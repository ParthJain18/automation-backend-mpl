import json
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar, Hashable
from langchain_core.tools import tool
from bs4 import BeautifulSoup
# Import Playwright
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from services.llm_config import get_llm_2
import time
import functools
import hashlib
import requests


class ExpiringCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            # Expired entry
            del self.cache[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (value, time.time())

    def clear(self) -> None:
        self.cache.clear()


page_cache = ExpiringCache(max_size=50, ttl_seconds=3600)
selector_cache = ExpiringCache(
    max_size=100, ttl_seconds=1800)


# Helper to generate cache keys from function arguments
def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from function arguments"""
    key_parts = []

    # Add positional args
    for arg in args:
        if isinstance(arg, (str, int, float, bool)) or arg is None:
            key_parts.append(str(arg))
        else:
            # For complex objects, use their string representation
            key_parts.append(str(hash(str(arg))))

    # Add keyword args (sorted for consistency)
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, (str, int, float, bool)) or v is None:
            key_parts.append(f"{k}={v}")
        else:
            # For complex objects, use their string representation
            key_parts.append(f"{k}={hash(str(v))}")

    # Create a hash of the combined string
    combined = "_".join(key_parts)
    return hashlib.md5(combined.encode()).hexdigest()


def _get_simplified_html_playwright(url: str, wait_time: int = 10000) -> str:
    """
    Fetches URL using Playwright, waits for render, extracts simplified HTML.
    Args:
        url (str): The URL to navigate to.
        wait_time (int): Max time in milliseconds to wait for page load events
                         before extracting HTML. Adjust as needed.
    """
    # Check cache first
    cache_key = generate_cache_key(url, wait_time)
    cached_result = page_cache.get(cache_key)
    if cached_result:
        print(f"Cache hit for URL: {url}")
        return cached_result

    print(f"Cache miss for URL: {url}, fetching fresh content")
    html_content = ""
    error_message = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=None)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            page = context.new_page()
            try:
                page.goto(url, timeout=wait_time,
                          wait_until='domcontentloaded')
                page.wait_for_timeout(1000)  # wait 1 second extra for JS
                html_content = page.content()
            except PlaywrightTimeoutError:
                error_message = f"Error: Timeout after {wait_time}ms waiting for page load at {url}."
                try:
                    html_content = page.content()
                except:
                    pass
            except Exception as e:
                error_message = f"Error: Playwright navigation/interaction failed for {url}: {e}"
                try:
                    html_content = page.content()
                except:
                    pass
            page.close()
            context.close()
            browser.close()
    except Exception as e:
        result = f"Error: Failed to initialize Playwright browser: {e}"
        # Don't cache errors from browser initialization
        return result

    if error_message and not html_content:
        # Don't cache complete failures
        return error_message

    try:
        soup = BeautifulSoup(html_content, 'lxml')
        allowed_tags = ['a', 'button', 'input', 'select', 'textarea',
                        'form', 'label', 'h1', 'h2', 'h3', 'h4', 'title', 'iframe', 'img']
        simplified_elements = []
        for element in soup.find_all(allowed_tags):
            attrs = {k: v for k, v in element.attrs.items() if k in [
                'id', 'class', 'name', 'type', 'placeholder', 'aria-label', 'role', 'value', 'href', 'action', 'method', 'src', 'alt', 'title']}
            text_content = element.get_text(strip=True)
            # Improve text extraction
            if not text_content:
                if element.name == 'input' and attrs.get('value'):
                    text_content = f"value: {attrs.get('value')}"
                elif element.name == 'input' and attrs.get('placeholder'):
                    text_content = f"placeholder: {attrs.get('placeholder')}"
                elif attrs.get('aria-label'):
                    text_content = f"aria-label: {attrs.get('aria-label')}"
                elif element.name == 'a' and attrs.get('href'):
                    text_content = f"href: {attrs.get('href')}"
                elif element.name == 'img' and attrs.get('alt'):
                    text_content = f"alt: {attrs.get('alt')}"
                elif element.name == 'img' and attrs.get('src'):
                    text_content = f"src: {attrs.get('src')}"
                elif attrs.get('title'):
                    text_content = f"title: {attrs.get('title')}"
            attr_str = ' '.join([f'{k}="{v}"' for k, v in attrs.items()])
            tag_repr = f"<{element.name} {attr_str}>{text_content}</{element.name}>"
            simplified_elements.append(tag_repr)

        MAX_LEN = 6000
        simplified_html = "\n".join(simplified_elements)
        if len(simplified_html) > MAX_LEN:
            simplified_html = simplified_html[:MAX_LEN] + "\n... (truncated)"

        if error_message:
            result = f"{error_message}\n\nAttempting to use potentially incomplete content:\n{simplified_html}"
        else:
            result = simplified_html

        # Cache the successful result
        page_cache.set(cache_key, result)
        return result
    except Exception as e:
        if error_message:
            result = f"{error_message}\n\nAdditionally, failed to parse the HTML content: {e}"
        else:
            result = f"Error: Content fetched from {url}, but failed during HTML simplification: {e}"
        # Don't cache parsing errors
        return result


plan_steps = []


@tool
def view_rendered_page(url: str) -> str:
    """
    (Uses Headless Browser) Fetches the content of a given URL after JavaScript rendering, waits for the page to load, and returns a simplified HTML view focusing on interactive elements. Use this to understand the structure of dynamic pages before deciding on interactions. Slower but more accurate for JS-heavy sites.
    """
    global plan_steps
    print(f"Tool: view_rendered_page called with URL: {url}")
    content = _get_simplified_html_playwright(url, wait_time=10000)
    print(
        f"Tool: view_rendered_page returning content summary (first 300 chars):\n{content[:300]}...")
    return content


@tool
def find_element_selector_llm(description: str, page_summary: str) -> str:
    """
    (Uses LLM) Analyzes the provided simplified HTML page summary to find the single best CSS selector for an element described by the user (e.g., 'the search input field', 'like button for the video', 'first link with text "Download"').
    Args:
        description (str): The natural language description of the element to find.
        page_summary (str): The simplified HTML content obtained from 'view_rendered_page'.
    Returns:
        str: The best CSS selector string, or an error message starting with 'ERROR_SELECTOR_GENERATION:'.
    """
    print(
        f"Tool: find_element_selector_llm called with description: '{description}'")

    # Basic validation
    if not description or not page_summary:
        return "ERROR_SELECTOR_GENERATION: Missing description or page_summary."
    if page_summary.startswith("Error:") or page_summary.startswith("ERROR_"):
        return f"ERROR_SELECTOR_GENERATION: Cannot find selector because the page_summary contains an error: {page_summary[:100]}..."

    # Check cache first
    # We use a truncated version of page_summary for the cache key to avoid huge keys
    # but still maintain uniqueness for the content
    page_summary_hash = hashlib.md5(page_summary.encode()).hexdigest()
    cache_key = generate_cache_key(description, page_summary_hash)
    cached_result = selector_cache.get(cache_key)
    if cached_result:
        print(f"Cache hit for selector: '{description}'")
        return cached_result

    print(f"Cache miss for selector: '{description}', generating with LLM")

    try:
        # Get a fresh LLM instance (using the config function)
        selector_llm = get_llm_2()

        # Define the prompt template for the selector generation task
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert web scraping and automation specialist. Your task is to identify the MOST RELIABLE CSS selector for specific elements based on HTML content and natural language descriptions.

                    SELECTOR GUIDELINES - FOLLOW PRECISELY:

                    1. PRIORITIZE selectors in this order:
                    - IDs: `#login-button` (most stable)
                    - Specific name attributes: `[name="q"]`, `[name="username"]`
                    - Unique aria attributes: `[aria-label="Search"]`, `[aria-describedby="instructions"]`
                    - Data attributes: `[data-testid="search-button"]`
                    - Specific input types: `input[type="submit"]`
                    - Classes WITH element types: `button.primary-action` NOT just `.primary-action`
                    - Combinations for uniqueness: `button.submit[type="submit"]`

                    2. AVOID these unreliable selectors:
                    - Generic class names alone: `.btn`, `.input`, `.card` (too common)
                    - Position-based selectors: `:nth-child(3)` (break easily)
                    - Complex nested paths: `div > div > span > a` (fragile)
                    - Tag names alone: `button`, `a`, `input` (too generic)

                    3. For AMBIGUOUS descriptions:
                    - Check if multiple elements match and choose most specific
                    - For elements with similar attributes, include multiple identifiers: `input[type="text"][placeholder="Search"]`
                    - If "first", "second", etc. specified, note this but prefer unique properties over position

                    4. For MULTIPLE elements:
                    - Return ONLY ONE selector that best matches the complete description
                    - If multiple distinct elements are requested, clearly state that multiple selectors are needed

                    RESPONSE FORMAT:
                    Return ONLY the CSS selector string with NO explanation, quotes, or formatting.
                    """
                ),
                (
                    "human",
                    """HTML Content:
                    ```html
                    {page_summary}
                    ```

                    Element Description: "{description}"

                    Return only the single best CSS selector for this element. No explanation, just the selector."""
                ),
            ]
        )

        chain = prompt | selector_llm | StrOutputParser()

        selector = chain.invoke({
            "page_summary": page_summary,
            "description": description
        })

        selector = selector.strip().strip('`"\'')
        if selector.lower().startswith("css"):
            selector = selector[3:].strip()

        # Extra cleaning for common formatting issues
        selector = selector.strip().replace('\n', '').replace('```', '')

        # Remove explanations that might appear after the selector
        if ' ' in selector and not (selector.startswith('[') and ']' in selector):
            potential_selector = selector.split(' ')[0]
            if potential_selector and (potential_selector.startswith('#') or
                                       potential_selector.startswith('.') or
                                       potential_selector.startswith('[') or
                                       '.' in potential_selector or
                                       '#' in potential_selector):
                print(
                    f"Cleaning selector from '{selector}' to '{potential_selector}'")
                selector = potential_selector

        print(
            f"Tool: find_element_selector_llm generated selector: '{selector}'")

        if not selector or "\n" in selector or " " in selector.strip() and not (selector.strip().startswith('[') and selector.strip().endswith(']')):
            print(
                f"Warning: Generated selector might be invalid or contain extra text: '{selector}'")
            if not selector:
                error = "ERROR_SELECTOR_GENERATION: LLM returned an empty selector."
                # Don't cache errors
                return error
            if " " in selector.strip() and not (selector.strip().startswith('[') and selector.strip().endswith(']')):
                is_complex_selector = any(c in selector for c in [
                                          '>', '+', '~']) or ' ' in selector
                if is_complex_selector:
                    print(
                        "Info: Generated selector is complex (contains spaces, >, +, or ~).")
                elif not (selector.startswith('[') and selector.endswith(']')):
                    error = f"ERROR_SELECTOR_GENERATION: LLM returned potentially invalid selector with spaces/newlines: '{selector}'"
                    # Don't cache errors
                    return error

        # Cache valid selectors
        selector_cache.set(cache_key, selector)
        return selector

    except Exception as e:
        print(f"Error during LLM call in find_element_selector_llm: {e}")
        import traceback
        traceback.print_exc()
        error = f"ERROR_SELECTOR_GENERATION: Exception occurred - {e}"
        # Don't cache errors
        return error


@tool
def add_action_step(name: str, type: str, **kwargs: Any) -> str:
    """
    Adds a single action step to the automation plan being built. Use this tool repeatedly to construct the sequence. Requires 'selector', 'value', 'url' etc. based on 'type'.
    Args:
        name (str): A descriptive name for the step (e.g., "Click Login Button").
        type (str): The type of action (e.g., 'click', 'input', 'newtab', 'wait', 'scroll', 'select', 'check', 'submit', 'collect', 'extract').
        **kwargs: Additional parameters required by the action type (e.g., 'selector', 'value', 'url'). Check the required fields for each action type.
    """
    global plan_steps
    print(
        f"Tool: add_action_step called: name='{name}', type='{type}', args={kwargs}")

    # Handle case where parameters might be nested inside a 'kwargs' dictionary
    if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
        nested_kwargs = kwargs.pop('kwargs')
        kwargs.update(nested_kwargs)

    required_fields = {
        "newtab": ["url"], "input": ["selector", "value"], "click": ["selector"],
        "wait": ["value"], "scroll": ["selector"], "select": ["selector", "value"],
        "check": ["selector"], "submit": ["selector"], "collect": ["selector"],
        "extract": ["selector"],
    }
    if type not in required_fields:
        return f"Error: Invalid action type '{type}'. Valid types are: {list(required_fields.keys())}"
    missing = [f for f in required_fields[type] if f not in kwargs]
    if missing:
        return f"Error: Missing required fields for type '{type}': {', '.join(missing)}. Provided: {list(kwargs.keys())}"

    if 'selector' in kwargs and isinstance(kwargs['selector'], str) and kwargs['selector'].startswith("ERROR_"):
        return f"Error: Cannot add step '{name}' because the provided selector indicates an error: {kwargs['selector']}"

    step = {"name": name, "type": type, **kwargs}
    if type == "wait" and isinstance(kwargs.get("value"), int):
        step["value"] = str(kwargs["value"])

    plan_steps.append(step)
    print(f"Tool: Step added to plan: {step}")
    return f"Step '{name}' ({type}) added. Plan has {len(plan_steps)} steps."


@tool
def finalize_plan(reason: str) -> str:
    """
    Call this ONLY when the user's request has been fully addressed by the steps added to the plan. Formats the collected steps into the final JSON output.
    """
    global plan_steps
    print(f"Tool: finalize_plan called. Reason: {reason}")
    if not plan_steps:
        return "Error: Cannot finalize an empty plan."
    for i, step in enumerate(plan_steps):
        if 'selector' in step and isinstance(step['selector'], str) and step['selector'].startswith("ERROR_"):
            return f"Error: Cannot finalize plan. Step {i+1} ('{step.get('name', 'Unnamed')}') contains an error selector: {step['selector']}"

    final_json = json.dumps(plan_steps, indent=2)
    print(f"Tool: finalize_plan returning JSON:\n{final_json}")
    return f"FINAL_PLAN_JSON:\n{final_json}"


@tool
def get_user_preference(query: str) -> str:
    """
    Retrieves user preferences and history information based on a natural language query.

    ONLY use this tool when you absolutely need user-specific information that isn't provided in the original request,
    such as favorite playlists, frequently visited forums, preferred settings, or browsing history patterns.

    Examples of when to use:
    - "What music genre does the user listen to most often?" when asked to play their favorite music
    - "What news sites does the user visit regularly?" when asked to check daily news
    - "What are the user's favorite YouTube channels?" when asked to show their favorite content

    DO NOT use for:
    - General information that's not user-specific
    - Information that was already provided in the initial request
    - When you can make reasonable generic assumptions (e.g., using YouTube's homepage instead of asking for preferences)

    Args:
        query (str): A specific natural language question about user preferences or history.
                     Be precise about what information you need (e.g., "What is the user's favorite YouTube channel?")

    Returns:
        str: User preference information relevant to the query, or error message if unavailable.
    """
    print(f"Tool: get_user_preference called with query: '{query}'")

    try:
        rag_endpoint = "https://prepared-tightly-mastiff.ngrok-free.app/user-query"

        try:
            response = requests.post(
                rag_endpoint,
                json={"query": query},
                timeout=10
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to user preference service: {e}"
            print(error_msg)
            return error_msg

        if response.status_code == 200:
            # Extract just the text response, ignoring any sources or screenshots
            response_data = response.json()

            # If the response is a tuple/list containing both response and sources
            if isinstance(response_data.get("response"), (list, tuple)) and len(response_data.get("response")) >= 1:
                # Get just the first item (text response) and ignore sources with screenshots
                result = response_data["response"][0]
            else:
                # Otherwise just get the response directly
                result = response_data.get("response", "No information found")

            # Ensure we're only returning text
            if isinstance(result, dict) and "sources" in result:
                # If the response is a dict with sources, extract just the text part
                result = result.get("text", str(result))

            print(f"Tool: get_user_preference returned: '{result}'")
            return str(result)
        else:
            error_msg = f"Error: Failed to retrieve user preferences. Status code: {response.status_code}"
            print(error_msg)
            return error_msg

    except Exception as e:
        error_msg = f"Error connecting to user preference service: {e}"
        print(error_msg)
        return error_msg


tools = [view_rendered_page, find_element_selector_llm,
         add_action_step, finalize_plan, get_user_preference]
