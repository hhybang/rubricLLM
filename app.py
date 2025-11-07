import streamlit as st
import os
import anthropic
from textwrap import dedent
import json
import re
import html
import time
from datetime import datetime
from pathlib import Path
from prompts import (
    COMPARE_WRITE_EDIT_PROMPT,
    RUBRIC_INFERENCE_SYSTEM_PROMPT,
    get_rubric_inference_user_prompt,
    CONTRASTIVE_TEXT_GENERATION_SYSTEM_PROMPT,
    get_contrastive_text_generation_user_prompt,
    PREFERENCE_ANALYSIS_SYSTEM_PROMPT,
    get_preference_analysis_user_prompt,
    get_comparison_prompt,
    build_system_instruction
)

# Project folder configuration
# Note: All projects are stored in the 'project' directory
# PROJECT_FOLDER is deprecated - use st.session_state.current_project instead
PROJECT_FOLDER = "op-ed"  # Default project name (for backward compatibility)

# Colors for criterion highlighting
CRITERION_COLORS = [
    "#FFB6C1",  # Light pink
    "#87CEEB",  # Sky blue
    "#90EE90",  # Light green
    "#F0E68C",  # Khaki
    "#FFA07A",  # Light salmon
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#FFCCCB",  # Light red
]


def parse_annotations(text):
    """
    Parse criterion annotations from text using simple number tags: <1>, <2>, etc.
    Returns: (clean_text, annotations_dict)
    where annotations_dict maps criterion_id -> list of (start, end, text) tuples
    """
    if not text:
        return text, {}
    
    # Check for both old format (<criterion_N>) and new format (<N>)
    has_format = re.search(r'<\d+>.*?</\d+>', text, re.DOTALL)
    
    if not (has_format):
        return text, {}
    
    # Determine which pattern to use
    if has_format:
        pattern = r'<(\d+)>(.*?)</\1>'
        tag_removal_pattern = r'<\d+>|</\d+>'
    
    # Remove all tags to get clean text first
    clean_text = re.sub(tag_removal_pattern, '', text)
    
    # Debug: save to file
    debug_path = Path("debug_annotations.txt")
    with open(debug_path, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Parsing Annotations ===\n")
        f.write(f"Clean text length: {len(clean_text)}\n")
        f.write(f"Clean text (first 200 chars): {repr(clean_text[:200])}\n")
    
    # Build a map of original position to clean position
    # We iterate through the text, tracking positions and skipping tags
    clean_pos_map = {}
    tag_positions = []
    
    # Find all tag positions
    for tag_match in re.finditer(tag_removal_pattern, text):
        tag_positions.append((tag_match.start(), tag_match.end()))
    
    # Now map positions: each character in text maps to its position in clean text
    clean_idx = 0
    for i in range(len(text)):
        # Check if this position is inside any tag
        in_tag = False
        for tag_start, tag_end in tag_positions:
            if tag_start <= i < tag_end:
                in_tag = True
                break
        
        if not in_tag:
            clean_pos_map[i] = clean_idx
            clean_idx += 1
    
    # Now find all annotation ranges using the position map
    annotations = {}
    
    # Find ALL tags (not just outermost) by finding all opening and closing tags
    # This handles nested tags like <6><4>text</4></6>
    all_tag_matches = []
    
    # Find all opening tags
    opening_pattern = r'<(\d+)>'
    for match in re.finditer(opening_pattern, text):
        tag_id = match.group(1)
        start_pos = match.start()
        # Find the matching closing tag for this opening tag
        closing_tag = f'</{tag_id}>'
        remaining_text = text[match.end():]
        
        # Track nested tags to find the correct closing tag
        depth = 1
        pos = 0
        while pos < len(remaining_text) and depth > 0:
            next_open = remaining_text.find(f'<{tag_id}>', pos)
            next_close = remaining_text.find(closing_tag, pos)
            
            if next_close == -1:
                break
            
            if next_open != -1 and next_open < next_close:
                # Found a nested opening tag
                depth += 1
                pos = next_open + len(f'<{tag_id}>')
            else:
                # Found a closing tag
                depth -= 1
                if depth == 0:
                    # Found the matching closing tag
                    end_pos = match.end() + next_close + len(closing_tag)
                    content_start = match.end()  # Start after opening tag
                    content_end = match.end() + next_close  # End before closing tag
                    
                    all_tag_matches.append({
                        'id': tag_id,
                        'start': start_pos,
                        'end': end_pos,
                        'content_start': content_start,
                        'content_end': content_end,
                        'content': remaining_text[:next_close]
                    })
                    break
                pos = next_close + len(closing_tag)
    
    # Now process each tag match to get positions in clean text
    for tag_match in all_tag_matches:
        criterion_id = tag_match['id']
        content_start = tag_match['content_start']
        content_end = tag_match['content_end']
        
        # Find the positions in clean_text by looking up in the position map
        start_pos = None
        end_pos = None
        
        # Find first and last characters of content in clean text
        for orig_pos in range(content_start, content_end):
            if orig_pos in clean_pos_map:
                if start_pos is None:
                    start_pos = clean_pos_map[orig_pos]
                end_pos = clean_pos_map[orig_pos] + 1
        
        if start_pos is not None and end_pos is not None:
            # Remove any nested tags from the content to get the actual text
            actual_content = re.sub(tag_removal_pattern, '', tag_match['content'])
            
            if criterion_id not in annotations:
                annotations[criterion_id] = []
            annotations[criterion_id].append((start_pos, end_pos, actual_content))
            
            # Debug: log this annotation
            with open(debug_path, 'a', encoding='utf-8') as f:
                f.write(f"\nCriterion {criterion_id}: positions {start_pos}-{end_pos}\n")
                f.write(f"  Raw content: {repr(tag_match['content'][:50])}\n")
                f.write(f"  Actual content (cleaned): {repr(actual_content[:50])}\n")
                f.write(f"  Text at positions: {repr(clean_text[start_pos:end_pos])}\n")
    
    # Debug: write final annotations
    with open(debug_path, 'a', encoding='utf-8') as f:
        f.write(f"\nFinal annotations: {annotations}\n")
        f.write(f"Clean text length: {len(clean_text)}\n")
    
    return clean_text, annotations

def highlight_text(text, annotations, active_criteria):
    """
    Highlight text based on active criteria.
    Returns HTML string with highlighted spans.
    
    IMPORTANT: Positions in annotations are relative to 'text' BEFORE escaping.
    We need to escape each fragment individually after slicing.
    """
    # Debug logging
    debug_path = Path("debug_highlight.txt")
    with open(debug_path, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Highlighting ===\n")
        f.write(f"Text length: {len(text)}\n")
        f.write(f"Text (first 200 chars): {repr(text[:200])}\n")
        f.write(f"Annotations: {annotations}\n")
        f.write(f"Active criteria: {active_criteria}\n")
    
    if not annotations or not active_criteria:
        return html.escape(text)
    
    # Get all annotations for active criteria
    highlights = []
    for criterion_id in active_criteria:
        if criterion_id in annotations:
            for start, end, content in annotations[criterion_id]:
                highlights.append((start, end, criterion_id))
    
    if not highlights:
        return html.escape(text)
    
    # Sort by position
    highlights.sort(key=lambda x: x[0])
    
    # Build highlighted HTML
    # CRITICAL: We slice from 'text' (unescaped) first, then escape each fragment
    result = []
    last_pos = 0
    
    for start, end, criterion_id in highlights:
        # Debug logging for each highlight
        with open(debug_path, 'a', encoding='utf-8') as f:
            f.write(f"\nHighlighting criterion {criterion_id} at positions {start}-{end}\n")
            f.write(f"  Text at these positions (unescaped): {repr(text[start:end])}\n")
        
        # Add text before highlight (escape after slicing)
        if start > last_pos:
            result.append(html.escape(text[last_pos:start]))
        
        # Get color for this criterion
        color_index = int(criterion_id) - 1 if criterion_id.isdigit() else 0
        color = CRITERION_COLORS[color_index % len(CRITERION_COLORS)]
        
        # Add highlighted span (escape after slicing)
        highlighted_content = html.escape(text[start:end])
        result.append(f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; border-left: 3px solid {color};">{highlighted_content}</span>')
        
        last_pos = end
    
    # Add remaining text (escape after slicing)
    if last_pos < len(text):
        result.append(html.escape(text[last_pos:]))
    
    return ''.join(result)

def display_rubric_criteria(rubric_data, container, comparison_rubric_data=None):
    """
    Display rubric criteria in a user-friendly format with headings, descriptions,
    point values, and expandable evidence sections. Criteria are grouped by category.

    If comparison_rubric_data is provided, highlights criteria that are added (new) or
    removed (missing in comparison) with different colors.
    """
    if not rubric_data or 'rubric' not in rubric_data:
        container.warning("No rubric data available")
        return

    rubric_list = rubric_data.get('rubric', [])

    if not rubric_list:
        container.info("No criteria defined")
        return

    # Build comparison map for efficient lookup
    comparison_map = {}
    if comparison_rubric_data and 'rubric' in comparison_rubric_data:
        for c in comparison_rubric_data['rubric']:
            name = c.get('name', '').lower().strip()
            comparison_map[name] = c

    # Group criteria by category
    from collections import defaultdict
    categories = defaultdict(list)
    for criterion in rubric_list:
        category = criterion.get('category', 'Uncategorized')
        # Add a flag to indicate if this is new or changed compared to comparison rubric
        if comparison_rubric_data:
            criterion_name = criterion.get('name', '').lower().strip()

            if criterion_name not in comparison_map:
                # Completely new criterion
                criterion['is_new'] = True
            else:
                # Check if description or points changed
                old_criterion = comparison_map[criterion_name]
                description_changed = criterion.get('description', '') != old_criterion.get('description', '')
                points_changed = criterion.get('points', 0) != old_criterion.get('points', 0)
                criterion['is_new'] = description_changed or points_changed
        categories[category].append(criterion)

    # Display each category group
    for category_name, criteria in categories.items():
        # Start the category box with HTML
        container.markdown(f"""
            <div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px; margin-bottom: 16px; background-color: rgba(33, 150, 243, 0.1);">
                <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #2196F3; text-transform: capitalize; text-align: center;">{category_name}</div>
        """, unsafe_allow_html=True)

        # Display criteria within this category
        for criterion in criteria:
            # Determine if this criterion is new (for highlighting)
            is_new = criterion.get('is_new', False)

            # Points value display
            points = criterion.get('points', 0)
            points_display = f"**Points: {points:+d}**" if points != 0 else "**Points: 0**"

            # Choose label based on whether it's new
            criterion_label = f"{criterion.get('name', 'Unnamed')}"
            if is_new:
                criterion_label = f"{'‼️'}  {criterion.get('name', 'Unnamed')}"

            with container.expander(criterion_label, expanded=False):
                # Description
                description = criterion.get('description', 'No description provided')
                st.markdown(f"{description}\n\n{points_display}")

                # Evidence section (expandable)
                evidence = criterion.get('evidence', [])
                if evidence:
                    with st.expander("📚 Evidence", expanded=False):
                        # Convert evidence list to single string
                        if isinstance(evidence, list):
                            evidence_text = ' '.join(str(item) for item in evidence)
                        else:
                            evidence_text = str(evidence)
                        st.markdown(evidence_text)
                else:
                    st.markdown("*No evidence examples provided*")

        # Close the category box
        container.markdown("</div>", unsafe_allow_html=True)

def _md_diff_to_html(marked_text):
    """
    Convert diff formatting to HTML with colors, processing markdown inline:
    +text+ → green addition span
    ~text~ → red deletion span
    Converts markdown **bold**, *italic*, etc to HTML.
    """
    if not marked_text:
        return ""
    
    # Process paragraphs separately
    paragraphs = marked_text.split("\n\n")
    
    result = "<div class='diff-container'>"
    
    for para in paragraphs:
        if not para.strip():
            continue
        
        # First escape HTML special characters
        para = (para
               .replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;"))
        
        # Process markdown formatting **bold**, *italic*
        para = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', para)
        para = re.sub(r'\*(.+?)\*', r'<em>\1</em>', para)
        
        # Then process diff markers
        # Process +text+ for additions (green)
        para = re.sub(r"\+(.+?)\+", r'<span class="diff-add">\1</span>', para)
        
        # Process ~text~ for deletions (red)  
        para = re.sub(r"~([^~]+)~", r'<span class="diff-del">\1</span>', para)
        
        result += f"<p>{para}</p>"
    
    result += "</div>"
    return result

def _parse_compare_output(text: str):
    """
    Extract Base, Rubric A, Rubric B, Key Differences, Summary from the model's combined output.
    We look for the exact headings requested in the prompt.
    Returns dict with keys: base, a, b, key_diffs, summary (strings).
    """
    import re
    s = text.replace("\r\n", "\n")

    # Key differences (before Stage 1)
    key_pat = r"###\s*Key Rubric Differences\s*\n(.*?)(?=\n###\s*Stage 1|\Z)"
    # Stage 1 – Base
    base_pat = r"###\s*Stage 1\s*–\s*Base Draft\s*\n(.*?)(?:\n###\s*Stage 2|\Z)"
    # Stage 2 – Revisions: A and B
    a_pat    = r"####\s*Rubric A Revision[^\n]*\n(.*?)(?:\n####\s*Rubric B Revision|\Z)"
    b_pat    = r"####\s*Rubric B Revision[^\n]*\n(.*?)(?:\n###\s*Summary|\n###\s*Stage 3|\Z)"
    # Summary (at end)
    sum_pat  = r"###\s*Summary of Impact\s*\n(.*)\Z"

    key_m  = re.search(key_pat,  s, flags=re.S)
    base_m = re.search(base_pat, s, flags=re.S)
    a_m    = re.search(a_pat,    s, flags=re.S)
    b_m    = re.search(b_pat,    s, flags=re.S)
    sum_m  = re.search(sum_pat,  s, flags=re.S)

    key_diffs = (key_m.group(1).strip() if key_m else "")
    base      = (base_m.group(1).strip() if base_m else "")
    a_txt     = (a_m.group(1).strip() if a_m else "")
    b_txt     = (b_m.group(1).strip() if b_m else "")
    summary   = (sum_m.group(1).strip() if sum_m else "")

    return {"base": base, "a": a_txt, "b": b_txt, "key_diffs": key_diffs, "summary": summary}

def _md_diff_to_html_compare(marked_text: str) -> str:
    """
    Convert **bold** → green add span, ~~strike~~ → red del span.
    Keep paragraphs. Works on the revision texts or comparison cells.
    """
    if marked_text is None:
        return ""
    # Escape HTML special chars first
    esc = (marked_text
           .replace("&", "&amp;")
           .replace("<", "&lt;")
           .replace(">", "&gt;"))
    # Bold additions
    esc = re.sub(r"\*\*(.+?)\*\*", r"<span class='add'>\1</span>", esc)
    # Strikethrough deletions
    esc = re.sub(r"~~(.+?)~~", r"<span class='del'>\1</span>", esc)
    # Paragraphs
    parts = [f"<p>{p}</p>" for p in esc.split("\n\n") if p.strip()]
    return "<div class='diff-wrap'>" + "".join(parts) + "</div>"

def compare_rubrics(task, rubric_a, rubric_b):
    """Compare two rubrics using the COMPARE_WRITE_EDIT_PROMPT"""
    prompt = COMPARE_WRITE_EDIT_PROMPT.format(
        task=task,
        rubric_a=json.dumps(rubric_a, indent=2),
        rubric_b=json.dumps(rubric_b, indent=2)
    )

    # Call Claude API directly
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )

    response = message.content[0].text

    # Parse using the exact same logic as the notebook
    parsed = _parse_compare_output(response)

    return {
        "base_txt": parsed["base"],
        "a_txt": parsed["a"],
        "b_txt": parsed["b"],
        "key_diffs": parsed["key_diffs"],
        "summary": parsed["summary"]
    }

# Set the API key
api_key = os.getenv('ANTHROPIC_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = api_key

client = anthropic.Anthropic()


def parse_analysis_and_content(full_text):
    """Extract analysis content and main content from response, removing both analysis and rubric_assessment tags"""
    # Pattern to match <analysis>...</analysis> tags
    analysis_pattern = r'<analysis>(.*?)</analysis>'
    analysis_matches = re.findall(analysis_pattern, full_text, re.DOTALL)

    # Extract analysis content
    analysis_content = '\n\n'.join(analysis_matches)

    # Remove complete analysis tags
    main_content = re.sub(analysis_pattern, '', full_text, flags=re.DOTALL)

    # Remove complete rubric_assessment tags
    rubric_assessment_pattern = r'<rubric_assessment>(.*?)</rubric_assessment>'
    main_content = re.sub(rubric_assessment_pattern, '', main_content, flags=re.DOTALL)

    # Handle incomplete opening tags (partial tags during streaming)
    if '<analysis>' in main_content:
        main_content = main_content.split('<analysis>')[0].strip()

    if '<rubric_assessment>' in main_content:
        main_content = main_content.split('<rubric_assessment>')[0].strip()

    return analysis_content, main_content

def parse_rubric_assessment(full_text):
    """Extract rubric assessment from response text
    Returns: dict of assessments by criterion name, or None if not found
    """
    # Pattern to match <rubric_assessment>...</rubric_assessment> tags
    pattern = r'<rubric_assessment>(.*?)</rubric_assessment>'
    match = re.search(pattern, full_text, re.DOTALL)

    if not match:
        return None

    assessment_content = match.group(1)

    # Extract JSON from code blocks
    json_pattern = r'```json\s*(.*?)\s*```'
    json_match = re.search(json_pattern, assessment_content, re.DOTALL)

    if not json_match:
        return None

    try:
        assessment_data = json.loads(json_match.group(1))
        # Convert list to dict keyed by criterion name for easy lookup
        assessments_by_name = {}
        for item in assessment_data.get('rubric_assessment', []):
            name = item.get('name')
            if name:
                assessments_by_name[name] = {
                    'score': item.get('score'),
                    'evidence': item.get('evidence'),
                    'areas_for_improvement': item.get('areas_for_improvement')
                }
        return assessments_by_name
    except json.JSONDecodeError:
        return None

def display_rubric_assessment(assessment_dict, message_id=None):
    """Display rubric assessment in a visually appealing format with feedback options"""
    if not assessment_dict:
        return

    st.markdown("---")

    # Initialize feedback storage in session state if needed
    if 'assessment_feedback' not in st.session_state:
        st.session_state.assessment_feedback = {}

    with st.expander("📊 Rubric Assessment", expanded=False):
        for idx, (criterion_name, data) in enumerate(assessment_dict.items()):
            score = data.get('score', 'N/A')
            evidence = data.get('evidence', '')
            areas_for_improvement = data.get('areas_for_improvement', '')

            # Create unique key for this criterion feedback
            feedback_key = f"{message_id}_{criterion_name}_{idx}" if message_id else f"{criterion_name}_{idx}"

            # Each criterion is also collapsible with score visible in header
            expander_label = f"{criterion_name} — Score: {score}"
            with st.expander(expander_label, expanded=False):
                # Create columns for content and feedback buttons
                col_content, col_feedback = st.columns([4, 1])

                with col_content:
                    st.markdown(f"**Evidence:** {evidence}")
                    st.markdown(f"**Areas for Improvement:** {areas_for_improvement}")

                with col_feedback:
                    st.markdown("**Feedback:**")
                    # Thumbs up/down buttons
                    col_up, col_down = st.columns(2)
                    with col_up:
                        if st.button("👍", key=f"thumbs_up_{feedback_key}", use_container_width=True):
                            if feedback_key not in st.session_state.assessment_feedback:
                                st.session_state.assessment_feedback[feedback_key] = {}
                            st.session_state.assessment_feedback[feedback_key]['rating'] = 'positive'
                            st.rerun()
                    with col_down:
                        if st.button("👎", key=f"thumbs_down_{feedback_key}", use_container_width=True):
                            if feedback_key not in st.session_state.assessment_feedback:
                                st.session_state.assessment_feedback[feedback_key] = {}
                            st.session_state.assessment_feedback[feedback_key]['rating'] = 'negative'
                            st.rerun()

                # Show current rating if exists
                if feedback_key in st.session_state.assessment_feedback:
                    rating = st.session_state.assessment_feedback[feedback_key].get('rating')
                    if rating:
                        rating_emoji = "👍" if rating == 'positive' else "👎"
                        st.caption(f"Your feedback: {rating_emoji}")

                # Optional text input for additional feedback
                text_feedback = st.text_area(
                    "Additional comments (optional):",
                    value=st.session_state.assessment_feedback.get(feedback_key, {}).get('comment', ''),
                    key=f"text_feedback_{feedback_key}",
                    placeholder="Share your thoughts on this assessment...",
                    height=80
                )

                # Store text feedback when it changes
                if text_feedback:
                    if feedback_key not in st.session_state.assessment_feedback:
                        st.session_state.assessment_feedback[feedback_key] = {}
                    st.session_state.assessment_feedback[feedback_key]['comment'] = text_feedback

def format_feedback_for_context():
    """Format collected assessment feedback into a structured message for the model"""
    if 'assessment_feedback' not in st.session_state or not st.session_state.assessment_feedback:
        return None

    # Get the most recent assistant message ID
    most_recent_message_id = None
    for msg in reversed(st.session_state.messages):
        if msg['role'] == 'assistant' and msg.get('message_id'):
            most_recent_message_id = msg['message_id']
            break

    if not most_recent_message_id:
        return None

    # Only collect feedback for the most recent assistant message
    feedback_items = []

    for feedback_key, feedback_data in st.session_state.assessment_feedback.items():
        # Only include feedback from the most recent message
        if not feedback_key.startswith(most_recent_message_id):
            continue

        # Parse the key to extract criterion name
        # Key format: "{message_id}_{criterion_name}_{idx}"
        parts = feedback_key.split('_')
        if len(parts) >= 4:
            # Skip message_id parts (assistant_{timestamp}), get everything except the last index
            criterion_name = '_'.join(parts[2:-1])
        else:
            criterion_name = feedback_key

        rating = feedback_data.get('rating', '')
        comment = feedback_data.get('comment', '')

        if rating or comment:
            feedback_str = f"- **{criterion_name}**:"
            if rating:
                feedback_str += f" {rating} feedback"
            if comment:
                feedback_str += f"\n  Comment: {comment}"
            feedback_items.append(feedback_str)

    if not feedback_items:
        return None

    feedback_message = """**[RUBRIC ASSESSMENT FEEDBACK]**
The user has provided feedback on your previous rubric assessment. Please incorporate this feedback into your response:

""" + "\n".join(feedback_items) + "\n\n**[USER'S REQUEST]**\n"

    return feedback_message

def stream_without_analysis(stream, response_placeholder, message_id):
    """Stream response while hiding analysis and rubric_assessment tags"""
    full_response = ""

    for text_chunk in stream.text_stream:
        full_response += text_chunk

        # Stop streaming if we hit rubric_assessment tag
        if '<rubric_assessment>' in full_response:
            # Get content before rubric_assessment tag
            content_before_assessment = full_response.split('<rubric_assessment>')[0]
            _, main_content = parse_analysis_and_content(content_before_assessment)
            response_placeholder.markdown(main_content)
            # Continue accumulating but don't display
            break

        # Parse the current accumulated response to filter out analysis
        _, main_content = parse_analysis_and_content(full_response)
        response_placeholder.markdown(main_content + "▌")

    # Continue reading the rest of the stream if we broke early
    for text_chunk in stream.text_stream:
        full_response += text_chunk

    # Final parse to ensure clean output
    analysis_content, main_content = parse_analysis_and_content(full_response)

    # Parse rubric assessment from the full response
    rubric_assessment = parse_rubric_assessment(full_response)

    if rubric_assessment:
        st.session_state.current_rubric_assessment = rubric_assessment

    # Parse annotations FROM THE MAIN CONTENT (after analysis is removed)
    # This ensures positions are correct
    clean_text, annotations = parse_annotations(main_content)

    # Store annotations with the rubric version that was active
    active_rubric_dict, active_idx, _ = get_active_rubric()
    rubric_version = active_rubric_dict.get('version', 1) if active_rubric_dict else None

    st.session_state.message_annotations[message_id] = {
        'clean_text': clean_text,
        'annotations': annotations,
        'original_main_content': main_content,  # Store this for reference
        'rubric_version': rubric_version  # Store the rubric version used for this message
    }

    # Display the clean text and assessment
    if rubric_assessment:
        # Create a container for both text and assessment
        with response_placeholder.container():
            st.markdown(clean_text)
            display_rubric_assessment(rubric_assessment, message_id)
    else:
        response_placeholder.markdown(clean_text)

    # Return the main content (with annotations but without analysis) and assessment for storage
    return main_content, analysis_content, rubric_assessment

def save_message_log(messages, rubric, analysis=None):
    """Save all messages to a log file"""
    # Create logs directory if it doesn't exist
    project_name = st.session_state.get('current_project', 'op-ed')
    log_dir = Path("project") / project_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"conversation-{timestamp}.json"
    
    # Prepare log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "rubric": rubric,
        "messages": messages,
        "analysis": analysis if analysis else ""
    }
    
    # Save to file
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    return log_file

# ========================
# Rubric Management Functions
# ========================
def get_rubric_file():
    """Get the current rubric file path based on active project"""
    project_name = st.session_state.get('current_project', 'op-ed')
    return Path("project") / project_name / "rubric_history.json"

def load_rubric_history():
    """Load rubric history from file"""
    try:
        rubric_file = get_rubric_file()
        if rubric_file.exists():
            with open(rubric_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading rubric history: {e}")
        return []

def save_rubric_history(history):
    """Save rubric history to file"""
    try:
        rubric_file = get_rubric_file()
        # Ensure parent directory exists
        rubric_file.parent.mkdir(parents=True, exist_ok=True)
        with open(rubric_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving rubric history: {e}")

def next_version_number():
    """Get the next version number for a new rubric"""
    hist = load_rubric_history()
    if not hist:
        return 1
    return max(r.get("version", 1) for r in hist) + 1

def get_active_rubric():
    """Get the active rubric and its index
    Returns: (full_rubric_dict, active_idx, rubric_history)
    where full_rubric_dict contains both 'version' and 'rubric' keys
    """
    hist = load_rubric_history()
    if not hist:
        return None, None, []
    
    idx = st.session_state.get("active_rubric_idx", len(hist) - 1)
    if 0 <= idx < len(hist):
        return hist[idx], idx, hist
    return hist[-1] if hist else None, len(hist) - 1 if hist else 0, hist

def infer_rubric_from_conversation(messages):
    """Infer a rubric from the conversation history using Claude"""
    # Build the conversation text for the prompt
    conversation_text = ""
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if role == 'user':
            conversation_text += f"\n\nUSER: {content}"
        elif role == 'assistant':
            conversation_text += f"\n\nASSISTANT: {content}"
    
    # Get the current active rubric to build upon
    active_rubric_dict, _, _ = get_active_rubric()
    previous_rubric = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
    previous_rubric_version = active_rubric_dict.get("version", 1) if active_rubric_dict else 1
    
    # Debug: Show if previous rubric is being used
    if previous_rubric:
        st.info(f"🔍 DEBUG: Using previous rubric v{previous_rubric_version}")
        previous_rubric_json = json.dumps(previous_rubric, ensure_ascii=False, indent=2)
        # previous_rubric_block = f"\n\nPREVIOUS RUBRIC (use this as a starting point and refine it based on the conversation):\n{previous_rubric_json}\n\nBuild upon this rubric by keeping what's relevant and adding/modifying criteria based on the conversation."
    else:
        st.info("🔍 DEBUG: No previous rubric found - creating new rubric from scratch")
        previous_rubric_json = ""
        # previous_rubric_block = "\n\nThis is a new rubric - create it from scratch based on the conversation."
    
    system_prompt = RUBRIC_INFERENCE_SYSTEM_PROMPT
    user_prompt = get_rubric_inference_user_prompt(conversation_text, previous_rubric_json)
        
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=20000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Extract the rubric JSON from the response
        response_text = response.content[0].text.strip()
        
        # Try to parse JSON from the response
        import re
        json_match = re.search(r'\{.*"rubric".*\}', response_text, re.DOTALL)
        if json_match:
            rubric_data = json.loads(json_match.group())
            # Add version number
            rubric_data["version"] = next_version_number()
            
            # Save to history
            hist = load_rubric_history()
            hist.append(rubric_data)
            save_rubric_history(hist)
            st.session_state.active_rubric_idx = len(hist) - 1
            
            return rubric_data
        else:
            rubric_data = json.loads(response_text)
            rubric_data["version"] = next_version_number()
            hist = load_rubric_history()
            hist.append(rubric_data)
            save_rubric_history(hist)
            st.session_state.active_rubric_idx = len(hist) - 1
            return rubric_data
    except Exception as e:
        st.error(f"Error inferring rubric: {str(e)}")
        return None

# Page configuration
st.set_page_config(
    page_title="AI Co-Writer",
    page_icon="✍️",
    layout="wide"
)

# Inject CSS styles for diff highlighting
st.markdown("""
<style>
.diff-container {
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.6;
}
.diff-add {
    background-color: #d4edda;
    color: #155724;
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 500;
}
.diff-del {
    background-color: #f8d7da;
    color: #721c24;
    text-decoration: line-through;
    padding: 2px 4px;
    border-radius: 3px;
}
.diff-container p {
    margin: 0.5em 0;
}

/* Keep chat input at bottom */
div[data-testid="chatInputContainer"] {
    position: fixed !important;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: white;
    padding: 1rem;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
}

/* Ensure chat input appears above all other content */
div[data-testid="chatInputContainer"] {
    z-index: 999 !important;
}

/* Add padding to main content area to prevent overlap */
main[data-testid="stMain"] {
    padding-bottom: 150px !important;
}

/* Also add padding to main block content */
.block-container {
    padding-bottom: 150px !important;
}

/* Ensure chat messages don't overlap the fixed input */
.stChatMessage {
    z-index: 1 !important;
}

/* Style for new criteria containers */
.criterion-container-new {
    border: 2px solid #FF9800 !important;
    border-radius: 6px !important;
    padding: 4px !important;
    margin-bottom: 8px !important;
    background-color: rgba(255, 152, 0, 0.1) !important;
}

.criterion-container-existing {
    border: 1px solid #2196F3 !important;
    border-radius: 6px !important;
    padding: 4px !important;
    margin-bottom: 8px !important;
    background-color: rgba(33, 150, 243, 0.05) !important;
}

/* Styles for rubric comparison diff highlighting */
.diff-wrap {
    font-family: monospace;
    line-height: 1.4;
}
.diff-wrap .add {
    background-color: #d4edda;
    color: #155724;
    padding: 1px 3px;
    border-radius: 2px;
    font-weight: bold;
}
.diff-wrap .del {
    background-color: #f8d7da;
    color: #721c24;
    text-decoration: line-through;
    padding: 1px 3px;
    border-radius: 2px;
}
.diff-wrap p {
    margin: 0.5em 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rubric' not in st.session_state:
    st.session_state.rubric = None

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = ""

if 'current_rubric_assessment' not in st.session_state:
    st.session_state.current_rubric_assessment = None

if 'selected_conversation' not in st.session_state:
    st.session_state.selected_conversation = None

if 'active_rubric_idx' not in st.session_state:
    hist = load_rubric_history()
    st.session_state.active_rubric_idx = len(hist) - 1 if hist else None

# Annotation management
if 'message_annotations' not in st.session_state:
    st.session_state.message_annotations = {}  # Maps message_id -> annotations dict

# Comparison mode
if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None  # Store regenerated response for comparison
if 'comparison_rubric_version' not in st.session_state:
    st.session_state.comparison_rubric_version = None  # Which rubric version was used for comparison

# Rubric comparison results (for Compare Rubrics tab)
if 'rubric_comparison_results' not in st.session_state:
    st.session_state.rubric_comparison_results = None

# Rubric assessment toggle (for enabling/disabling rubric assessment in responses)
if 'rubric_assessment_enabled' not in st.session_state:
    st.session_state.rubric_assessment_enabled = True  # Default to enabled

# Initialize rubric from active version if not set
if st.session_state.rubric is None and st.session_state.active_rubric_idx is not None:
    active_rubric_dict, _, _ = get_active_rubric()
    if active_rubric_dict:
        rubric_list = active_rubric_dict.get("rubric", [])
        st.session_state.rubric = rubric_list

def simple_markdown_to_html(text):
    """Convert simple markdown formatting to HTML"""
    import re
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    # Italic: *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
    # Line breaks
    text = text.replace('\n', '<br>')
    return text

def generate_text_pair(writing_task, dimension, base_rubric=None, criteria_description=None):
    """
    Generate a pair of contrasting texts for preference comparison.

    Args:
        writing_task: The writing task description
        dimension: The rubric dimension/criterion name to focus on
        base_rubric: Optional existing rubric to inform generation
        criteria_description: Specific description of the criterion

    Returns:
        tuple: (text_a, text_b, explanation_a, explanation_b) - two contrasting text samples with explanations
    """
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    system_prompt = CONTRASTIVE_TEXT_GENERATION_SYSTEM_PROMPT
    prompt = get_contrastive_text_generation_user_prompt(writing_task, dimension, criteria_description, base_rubric)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            system = system_prompt,
            max_tokens=20000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text

        # Parse the response to extract Text A, B and Explanations
        text_a = ""
        text_b = ""
        explanation_a = ""
        explanation_b = ""

        if "TEXT A:" in response_text and "TEXT B:" in response_text:
            # Split by TEXT B first
            parts = response_text.split("TEXT B:")
            part_a = parts[0]
            part_b = parts[1] if len(parts) > 1 else ""

            # Extract TEXT A and EXPLANATION A
            if "EXPLANATION A:" in part_a:
                a_parts = part_a.split("EXPLANATION A:")
                text_a = a_parts[0].replace("TEXT A:", "").strip()
                explanation_a = a_parts[1].strip()
            else:
                text_a = part_a.replace("TEXT A:", "").strip()

            # Extract TEXT B and EXPLANATION B
            if "EXPLANATION B:" in part_b:
                b_parts = part_b.split("EXPLANATION B:")
                text_b = b_parts[0].strip()
                explanation_b = b_parts[1].strip()
            else:
                text_b = part_b.strip()

            return text_a, text_b, explanation_a, explanation_b
        else:
            # Fallback if parsing fails
            return (
                f"[Generated text A for {dimension}]\n\n{response_text[:len(response_text)//2]}",
                f"[Generated text B for {dimension}]\n\n{response_text[len(response_text)//2:]}",
                "",
                ""
            )
    except Exception as e:
        st.error(f"Error generating text pair: {e}")
        return (
            f"Error generating text A for {dimension}",
            f"Error generating text B for {dimension}",
            "",
            ""
        )

def analyze_preferences_and_generate_rubric(preferences, writing_task, criteria_info, base_rubric=None):
    """
    Analyze user preferences and generate a refined rubric.

    Args:
        preferences: List of preference dictionaries
        writing_task: The writing task description
        criteria_info: Dict with criterion details (name, category, description, points)
        base_rubric: Optional existing rubric to refine

    Returns:
        dict: Generated rubric structure
    """
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    # Build preference summary
    pref_summary = []
    for pref in preferences:
        chosen_text = pref['text_a'] if pref['choice'] == 'a' else pref['text_b']
        rejected_text = pref['text_b'] if pref['choice'] == 'a' else pref['text_a']
        pref_summary.append(f"""
            Preferred text: {chosen_text[:200]}...
            Rejected text: {rejected_text[:200]}...
            """)
    # print (pref_summary)

    # Extract criterion details
    criterion_name = criteria_info.get('name', 'Unknown')
    criterion_category = criteria_info.get('category', 'General')
    criterion_description = criteria_info.get('description', '')
    criterion_points = criteria_info.get('points', 0)

    system_prompt = PREFERENCE_ANALYSIS_SYSTEM_PROMPT
    prompt = get_preference_analysis_user_prompt(
        writing_task, criterion_name, criterion_category,
        criterion_description, criterion_points, pref_summary
    )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=20000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text
        # print(f"API Response: {response_text}")  # Debug print

        # Extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text.strip()

        # print(f"Extracted JSON: {json_str}")  # Debug print

        if not json_str:
            st.error("Empty response from API")
            return None

        rubric_data = json.loads(json_str)
        return rubric_data
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {e}")
        st.error(f"Response was: {response_text if 'response_text' in locals() else 'No response'}")
        return None
    except Exception as e:
        st.error(f"Error generating rubric: {e}")
        return None

def load_conversations():
    """Load all conversation files from logs directory"""
    project_name = st.session_state.get('current_project', 'op-ed')
    log_dir = Path("project") / project_name / "logs"
    if not log_dir.exists():
        return []
    
    conversations = []
    for file in sorted(log_dir.glob("conversation-*.json"), reverse=True):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract readable date from filename
                filename = file.stem
                date_str = filename.replace("conversation-", "")
                conversations.append({
                    "filename": file.name,
                    "timestamp": data.get("timestamp", date_str),
                    "messages_count": len(data.get("messages", [])),
                    "data": data
                })
        except Exception as e:
            continue
    return conversations

def load_conversation_data(filepath):
    """Load conversation data from file"""
    project_name = st.session_state.get('current_project', 'op-ed')
    log_dir = Path("project") / project_name / "logs"
    file = log_dir / filepath
    if file.exists():
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_available_projects():
    """Get list of available project folders from the 'project' directory"""
    projects_dir = Path("project")
    if not projects_dir.exists():
        projects_dir.mkdir(exist_ok=True)
        return []

    projects = []
    for item in projects_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            projects.append(item.name)
    return sorted(projects)

def create_new_project(project_name):
    """Create a new project folder with necessary structure inside the 'project' directory"""
    # Validate project name (alphanumeric, hyphens, underscores only)
    import string
    valid_chars = string.ascii_letters + string.digits + '-_'
    if not all(c in valid_chars for c in project_name):
        return False, "Project name can only contain letters, numbers, hyphens, and underscores"

    # Ensure project directory exists
    projects_dir = Path("project")
    projects_dir.mkdir(exist_ok=True)

    project_path = projects_dir / project_name
    if project_path.exists():
        return False, "Project folder already exists"

    try:
        # Create project folder
        project_path.mkdir(exist_ok=True)

        # Create logs subfolder
        logs_path = project_path / "logs"
        logs_path.mkdir(exist_ok=True)

        # Create empty rubric_history.json
        rubric_file = project_path / "rubric_history.json"
        with open(rubric_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)

        return True, f"Project '{project_name}' created successfully!"
    except Exception as e:
        return False, f"Error creating project: {str(e)}"

# Title at the top
st.title("✍️ AI-Rubric Writer")
st.markdown("Collaborate with AI to improve your writing!")

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Compare Rubrics", "📝 Build Rubric"])

with tab1:
    conversations = load_conversations()

    # Create selection with conversation details
    conversation_options = [("New Conversation", None)]
    if conversations:
        for conv in conversations:
            # Format timestamp for display
            try:
                dt = datetime.fromisoformat(conv["timestamp"])
                formatted_time = dt.strftime("%m/%d %H:%M")
                display = f"{formatted_time} ({conv['messages_count']} msgs)"
            except:
                display = f"{conv['timestamp']} ({conv['messages_count']} msgs)"
            conversation_options.append((display, conv["filename"]))

    # Create selectbox with current selection
    current_selection = st.session_state.selected_conversation or None
    options = [opt[1] for opt in conversation_options]

    # Find index of current selection
    current_index = 0
    if current_selection:
        try:
            current_index = options.index(current_selection)
        except ValueError:
            current_index = 0

    selected_file = st.selectbox(
        "💬 Select conversation:",
        options=options,
        format_func=lambda x: next(opt[0] for opt in conversation_options if opt[1] == x) if x is not None else "New Conversation",
        index=current_index,
        key="conversation_selector"
    )
    
    # Clear the just_saved flag if we're not transitioning conversations
    if selected_file == st.session_state.get("selected_conversation") and st.session_state.get("just_saved"):
        st.session_state.just_saved = None
    
    # Handle "New Conversation" selection
    if selected_file is None and st.session_state.selected_conversation is not None:
        # Explicitly switching FROM a loaded conversation TO "New Conversation"
        st.session_state.messages = []
        st.session_state.rubric = None
        st.session_state.current_analysis = ""
        st.session_state.selected_conversation = None
        st.session_state.message_annotations = {}
        st.session_state.comparison_result = None
        st.session_state.comparison_rubric_version = None
        st.rerun()
    elif selected_file and selected_file != st.session_state.selected_conversation:
        # Check if we just saved this conversation - if so, don't reload it
        just_saved = st.session_state.get("just_saved")
        if just_saved and selected_file == just_saved:
            # We just saved this conversation, so keep the current messages
            st.session_state.selected_conversation = selected_file
            st.session_state.just_saved = None  # Clear the flag
        else:
            # Load the selected conversation
            conv_data = load_conversation_data(selected_file)
            if conv_data:
                st.session_state.messages = conv_data.get("messages", [])
                st.session_state.rubric = conv_data.get("rubric", None)
                st.session_state.current_analysis = conv_data.get("analysis", "")  # Load saved analysis
                st.session_state.selected_conversation = selected_file
                st.rerun()
    
    st.divider()
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        if message['role'] == 'system':
            st.info(message['content'])
        else:
            with st.chat_message(message['role']):
                # Get message_id - prefer stored one, otherwise generate
                message_id = message.get('message_id', f"{message['role']}_{idx}")

                # For user messages, always show the full content (which includes feedback)
                # For assistant messages, use display_content (clean version without annotations)
                if message['role'] == 'user':
                    content_to_display = message['content']
                else:
                    content_to_display = message.get('display_content', message['content'])

                if message_id in st.session_state.message_annotations:
                    # Get annotation data
                    ann_data = st.session_state.message_annotations[message_id]
                    clean_text = ann_data['clean_text']

                    # Just display the clean text without any highlighting
                    st.markdown(clean_text)

                    # Display rubric assessment if available
                    if message.get('rubric_assessment'):
                        display_rubric_assessment(message['rubric_assessment'], message_id)
                else:
                    # For backward compatibility, try to parse if not already parsed
                    # Check for both new format (<N>) and old format (<criterion_N>)
                    has_new_format = bool(re.search(r'<\d+>.*?</\d+>', message['content'], re.DOTALL))
                    has_old_format = '<criterion_' in message['content']

                    if has_new_format or has_old_format:
                        # Remove analysis first if present, then parse annotations
                        analysis_pattern = r'<analysis>(.*?)</analysis>'
                        content_without_analysis = re.sub(analysis_pattern, '', message['content'], flags=re.DOTALL)

                        # Now parse annotations
                        clean_text, annotations = parse_annotations(content_without_analysis)
                        if annotations:
                            # Store in session state
                            # Try to get rubric version from the message or default to None
                            rubric_version = message.get('rubric_version', None)

                            st.session_state.message_annotations[message_id] = {
                                'clean_text': clean_text,
                                'annotations': annotations,
                                'rubric_version': rubric_version
                            }

                            # Update display content to match our parsed clean_text
                            message['display_content'] = clean_text

                            # Just display the clean text without any highlighting
                            st.markdown(clean_text)

                            # Display rubric assessment if available
                            if message.get('rubric_assessment'):
                                display_rubric_assessment(message['rubric_assessment'], message_id)
                        else:
                            # No annotations found, display as-is
                            st.markdown(content_to_display)

                            # Display rubric assessment if available
                            if message.get('rubric_assessment'):
                                display_rubric_assessment(message['rubric_assessment'], message_id)
                    else:
                        # No annotation tags, display as-is
                        st.markdown(content_to_display)

                        # Display rubric assessment if available
                        if message.get('rubric_assessment'):
                            display_rubric_assessment(message['rubric_assessment'], message_id)
    
    # Display comparison result if it exists
    if st.session_state.comparison_result:
        st.divider()
        col_title, col_close = st.columns([4, 1])
        
        with col_title:
            st.markdown("### ⚖️ Comparison Response")
        
        with col_close:
            if st.button("✖️ Close", key="close_comparison", use_container_width=True):
                st.session_state.comparison_result = None
                st.session_state.comparison_rubric_version = None
                st.rerun()
        
        comp_result = st.session_state.comparison_result
        comp_message_id = comp_result['message_id']
        clean_text = comp_result['clean_text']
        annotations = comp_result['annotations']
        
        # Show which rubric version was used
        comp_rubric_version = st.session_state.comparison_rubric_version
        st.caption(f"Response regenerated with Rubric v{comp_rubric_version}")
        
        # Get active criteria for comparison highlighting (show all annotations)
        comp_active_criteria = list(annotations.keys()) if annotations else []
        
        # Check if we should show version mismatch warning
        active_rubric_dict, active_idx, _ = get_active_rubric()
        current_rubric_version = active_rubric_dict.get('version', 1) if active_rubric_dict else None
        should_highlight_comparison = (comp_rubric_version == current_rubric_version)
        
        # Display without highlighting
        with st.chat_message("assistant"):
            # Check for diff markers
            has_diff_markers = re.search(r'\+[^+]+\+', clean_text) or re.search(r'~[^~]+~', clean_text)

        if has_diff_markers:
            # Show diff highlighting
            diff_html = _md_diff_to_html(clean_text)
            st.markdown(diff_html, unsafe_allow_html=True)
        else:
            # Just show the clean text
            st.markdown(clean_text)

    # Toggle for Rubric Assessment (above chat input)
    st.session_state.rubric_assessment_enabled = st.checkbox(
        "📊 Rubric Assessment",
        value=st.session_state.rubric_assessment_enabled,
        help="When enabled, the model will provide a rubric assessment with scores for each criterion in drafts and revisions"
    )

    # User input (chat input and buttons)
    if prompt := st.chat_input("Type your message here..."):
        # Clear comparison when starting a new message
        st.session_state.comparison_result = None
        st.session_state.comparison_rubric_version = None

        # Check if there's feedback to incorporate
        feedback_context = format_feedback_for_context()

        # Add rubric assessment reminder if enabled
        assessment_reminder = ""
        if st.session_state.rubric_assessment_enabled:
            active_rubric_dict, _, _ = get_active_rubric()
            if active_rubric_dict and active_rubric_dict.get("rubric"):
                assessment_reminder = "\n\n[REMINDER: Please include a <rubric_assessment> section at the end of your response with scores for each rubric criterion.]"

        # Prepend feedback to user's message if available, append assessment reminder
        if feedback_context:
            full_message = feedback_context + prompt + assessment_reminder
            # Store both the full message (for API) and display version (for UI)
            st.session_state.messages.append({
                "role": "user",
                "content": full_message,  # Full message with feedback for API
                "display_content": prompt  # Just the user's prompt for display
            })

            # Clear the feedback after incorporating it
            st.session_state.assessment_feedback = {}
        else:
            # No feedback, but add assessment reminder if enabled
            full_message = prompt + assessment_reminder
            st.session_state.messages.append({"role": "user", "content": full_message})

        # Display user message (show the full message with feedback if present)
        with st.chat_message("user"):
            if feedback_context:
                # Show the full message with feedback context
                st.markdown(full_message)
            else:
                st.markdown(prompt)
        
        # Prepare message history for API
        # Use original content (with annotations) for context, but we'll extract for API
        api_messages = []
        for msg in st.session_state.messages:
            # Use the original content (which may have annotations) for API context
            # This ensures the LLM sees its own annotation pattern in conversation history
            content_to_send = msg.get('content', msg.get('display_content', ''))
            api_messages.append({
                "role": msg['role'],
                "content": content_to_send
            })
        
        # Show assistant response with streaming
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_placeholder = st.empty()
                
                # Generate unique message ID for this response (use timestamp to ensure uniqueness)
                import time
                message_id = f"assistant_{int(time.time() * 1000000)}"
                
                # Get the current active rubric for the system prompt
                active_rubric_dict, _, _ = get_active_rubric()
                active_rubric_list = active_rubric_dict.get("rubric", []) if active_rubric_dict else []

                # Build system instruction (use the toggle state for assessment)
                system_instruction = build_system_instruction(
                    active_rubric_list,
                    include_assessment=st.session_state.rubric_assessment_enabled
                )

                # Debug: Show which rubric is being used
                if active_rubric_list:
                    assessment_status = "enabled" if st.session_state.rubric_assessment_enabled else "disabled"
                    st.caption(f"🔍 Using rubric with {len(active_rubric_list)} criteria (Assessment: {assessment_status})")
                else:
                    st.caption("🔍 No rubric active - system instruction will not include rubric")

                # Stream the response
                try:
                    with client.messages.stream(
                        max_tokens=20000,
                        system=system_instruction,
                        messages=api_messages,
                        model="claude-sonnet-4-5",
                    ) as stream:
                        # Stream and filter out analysis tags in real-time
                        # Returns: (main_content with annotations but no analysis, analysis_content, rubric_assessment)
                        main_content, analysis_content, rubric_assessment = stream_without_analysis(stream, response_placeholder, message_id)

                    # Get the clean text and annotated content
                    clean_text_to_store = ""
                    original_with_annotations = main_content  # main_content has annotations but no analysis
                    
                    if message_id in st.session_state.message_annotations:
                        clean_text_to_store = st.session_state.message_annotations[message_id]['clean_text']
                        # Use main_content which has annotations
                        original_with_annotations = main_content
                    else:
                        # Fallback - parse annotations from main_content
                        clean_text_to_store, _ = parse_annotations(main_content)
                        if not clean_text_to_store:
                            clean_text_to_store = main_content
                        original_with_annotations = main_content
                    
                    # Get the currently active rubric version to store with the message
                    active_rubric_dict, active_idx, _ = get_active_rubric()
                    rubric_version = active_rubric_dict.get('version', 1) if active_rubric_dict else None
                    
                    # Store message with BOTH annotated and clean versions
                    # The 'content' field stores the original with annotations for API context
                    # The display logic will show the clean version
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": original_with_annotations,  # Store WITH annotations for context
                        "display_content": clean_text_to_store,  # Store clean version for display
                        "message_id": message_id,
                        "rubric_version": rubric_version,  # Store which rubric version was used
                        "rubric_assessment": rubric_assessment  # Store the assessment data
                    })
                    
                    # Update analysis in session state and rerun to show in sidebar
                    st.session_state.current_analysis = analysis_content
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error occurred: {str(e)}")
    
    # Buttons below chat input
    if st.session_state.messages:
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            save_button = st.button("💾 Save Conversation", use_container_width=True)
            if save_button:
                try:
                    # Save the conversation to a file
                    log_file = save_message_log(st.session_state.messages, st.session_state.rubric, st.session_state.current_analysis)
                    
                    # Store a flag to indicate we just saved (don't reload)
                    st.session_state.just_saved = log_file.name
                    
                    # Update selected_conversation to the saved file
                    st.session_state.selected_conversation = log_file.name
                    st.success(f"✓ Log saved to {log_file.name}")
                    st.balloons()  # Show celebration
                    st.rerun()  # Rerun to update the selector
                except Exception as e:
                    st.error(f"Error saving log: {str(e)}")
        
        with btn_col2:
            infer_button = st.button("🔍 Infer Rubric", use_container_width=True)
            if infer_button:
                if not st.session_state.messages:
                    st.error("No conversation to infer rubric from!")
                else:
                    # Store key before inference
                    preserve_state = {
                        'messages': st.session_state.messages.copy(),
                        'selected_conversation': st.session_state.selected_conversation,
                        'current_analysis': st.session_state.current_analysis
                    }
                    
                    with st.spinner("Inferring rubric from conversation..."):
                        inferred_rubric_data = infer_rubric_from_conversation(preserve_state['messages'])
                        if inferred_rubric_data:
                            # Restore preserved state
                            st.session_state.messages = preserve_state['messages']
                            st.session_state.selected_conversation = preserve_state['selected_conversation']
                            st.session_state.current_analysis = preserve_state['current_analysis']
                            
                            # Update rubric and editing criteria
                            st.session_state.rubric = inferred_rubric_data.get("rubric")
                            
                            # Get the newly set active rubric (which should now be the inferred one)
                            active_rubric_dict, active_idx, _ = get_active_rubric()
                            if active_rubric_dict:
                                # active_rubric_dict is a dict with "rubric" and "version" fields
                                rubric_list = active_rubric_dict.get("rubric", [])
                                st.session_state.editing_criteria = rubric_list.copy()
                            
                            st.success(f"✓ Rubric inferred (v{inferred_rubric_data.get('version', '?')})")
                            st.rerun()  # Rerun to update the sidebar
                        else:
                            # Restore preserved state on failure
                            st.session_state.messages = preserve_state['messages']
                            st.session_state.selected_conversation = preserve_state['selected_conversation']
                            st.session_state.current_analysis = preserve_state['current_analysis']
                            st.error("Failed to infer rubric")
        
        with btn_col3:
            clear_button = st.button("🗑️ Clear Conversation", use_container_width=True)
            if clear_button:
                st.session_state.messages = []
                st.session_state.current_analysis = ""
                st.session_state.selected_conversation = None
                st.session_state.comparison_result = None
                st.session_state.comparison_rubric_version = None
                st.rerun()
    
    # Comparison mode UI (only show if there are messages and an assistant response)
    if st.session_state.messages and any(msg['role'] == 'assistant' for msg in st.session_state.messages):
        st.divider()
        col_left, col_right = st.columns([3, 1])
        
        with col_left:
            # Get rubric history for comparison
            _, _, rubric_history = get_active_rubric()
            if rubric_history and len(rubric_history) > 1:
                comparison_options = [f"v{r.get('version', 1)}" for r in rubric_history]
                selected_comparison = st.selectbox(
                    "🔍 Compare with Rubric Version:",
                    options=comparison_options,
                    key="comparison_rubric_selector",
                    help="Select a different rubric version to see how it would affect the last response"
                )
        
        with col_right:
            compare_button = st.button("⚖️ Compare", use_container_width=True, key="compare_button")
            if compare_button:
                # Get the last user message
                last_assistant_idx = None
                last_user_message = None
                
                # Find the last assistant message and the user message before it
                for i in range(len(st.session_state.messages) - 1, -1, -1):
                    if st.session_state.messages[i]['role'] == 'assistant':
                        last_assistant_idx = i
                        # Find the user message before this assistant
                        for j in range(i - 1, -1, -1):
                            if st.session_state.messages[j]['role'] == 'user':
                                last_user_message = st.session_state.messages[j]['content']
                                break
                        break
                
                if last_user_message and selected_comparison:
                    # Get the rubric to compare with
                    compare_idx = int(selected_comparison.replace('v', '')) - 1
                    if 0 <= compare_idx < len(rubric_history):
                        compare_rubric_dict = rubric_history[compare_idx]
                        compare_rubric_list = compare_rubric_dict.get("rubric", [])
                        
                        # Store which rubric version we're comparing with
                        st.session_state.comparison_rubric_version = compare_rubric_dict.get("version", compare_idx + 1)
                        
                        # Get the last assistant response to reference
                        last_assistant_message = st.session_state.messages[last_assistant_idx]
                        last_assistant_content = last_assistant_message.get('display_content', last_assistant_message.get('content', ''))
                        
                        # Prepare a prompt asking how the draft should change with this rubric
                        comparison_prompt = get_comparison_prompt(
                            last_assistant_content,
                            st.session_state.comparison_rubric_version,
                            compare_rubric_list
                        )
                        
                        # Generate revision suggestions with the comparison rubric
                        with st.spinner(f"Analyzing with {selected_comparison}..."):
                            try:
                                import time
                                comparison_message_id = f"comparison_{int(time.time() * 1000000)}"
                                
                                # Use the comparison rubric for the system instruction (use current toggle state)
                                response = client.messages.create(
                                    max_tokens=20000,
                                    system=build_system_instruction(
                                        compare_rubric_list,
                                        include_assessment=st.session_state.rubric_assessment_enabled
                                    ),
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": comparison_prompt
                                        }
                                    ],
                                    model="claude-sonnet-4-5",
                                )
                                
                                # Parse the response
                                full_response = response.content[0].text
                                
                                # Remove analysis tags and parse annotations
                                analysis_content, main_content = parse_analysis_and_content(full_response)
                                clean_text, annotations = parse_annotations(main_content)
                                
                                # Store comparison result
                                st.session_state.comparison_result = {
                                    'clean_text': clean_text,
                                    'annotations': annotations,
                                    'main_content': main_content,
                                    'analysis_content': analysis_content,
                                    'message_id': comparison_message_id
                                }
                                
                                # Store in message_annotations for highlighting to work
                                st.session_state.message_annotations[comparison_message_id] = {
                                    'clean_text': clean_text,
                                    'annotations': annotations,
                                    'rubric_version': st.session_state.comparison_rubric_version
                                }
                                
                                st.success(f"✓ Revision suggestions generated with {selected_comparison}")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error during comparison: {str(e)}")
                    else:
                        st.error("Invalid rubric version selected")
                else:
                    st.error("Unable to find previous user message for comparison")

# Sidebar for rubric input
with st.sidebar:
    # Project Selector at the top
    st.header("📁 Project")

    # Get available projects
    available_projects = get_available_projects()

    # Initialize project in session state if not exists
    if 'current_project' not in st.session_state:
        st.session_state.current_project = 'op-ed'  # Default project name

    # Project selector
    if available_projects:
        selected_project = st.selectbox(
            "Select Project:",
            options=available_projects,
            index=available_projects.index(st.session_state.current_project) if st.session_state.current_project in available_projects else 0,
            key="project_selector"
        )

        # Update current_project if selection changed
        if selected_project != st.session_state.current_project:
            st.session_state.current_project = selected_project
            # Reset session state
            st.session_state.messages = []
            st.session_state.selected_conversation = None
            if 'active_rubric_idx' in st.session_state:
                del st.session_state.active_rubric_idx

            # Load the new project's rubric
            active_rubric_dict, active_idx, _ = get_active_rubric()
            if active_rubric_dict and active_rubric_dict.get("rubric"):
                rubric_list = active_rubric_dict.get("rubric", [])
                st.session_state.rubric = rubric_list
                st.session_state.editing_criteria = rubric_list.copy()
            else:
                st.session_state.rubric = []
                st.session_state.editing_criteria = []

            st.rerun()
    else:
        st.info("No projects found. Create one below!")

    # Create new project section
    # Initialize expander state if not exists
    if 'create_project_expanded' not in st.session_state:
        st.session_state.create_project_expanded = False

    with st.expander("➕ Create New Project", expanded=st.session_state.create_project_expanded):
        new_project_name = st.text_input(
            "Project Name:",
            placeholder="e.g., my-essay-project",
            key="new_project_name"
        )

        if st.button("Create Project", use_container_width=True):
            if new_project_name.strip():
                success, message = create_new_project(new_project_name.strip())
                if success:
                    st.success(message)
                    # Update current project
                    st.session_state.current_project = new_project_name.strip()
                    # Reset session state
                    st.session_state.rubric = []
                    st.session_state.editing_criteria = []
                    st.session_state.messages = []
                    st.session_state.selected_conversation = None
                    # Clear the text input and collapse expander
                    if 'new_project_name' in st.session_state:
                        del st.session_state.new_project_name
                    st.session_state.create_project_expanded = False
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter a project name")

    st.divider()

    st.header("📋 Rubric Configuration")

    # Get active rubric
    active_rubric_dict, active_idx, rubric_history = get_active_rubric()
    
    # Initialize editing criteria if needed
    if "editing_criteria" not in st.session_state:
        if active_rubric_dict:
            rubric_list = active_rubric_dict.get("rubric", [])
            st.session_state.editing_criteria = rubric_list.copy() if rubric_list else []
        else:
            st.session_state.editing_criteria = []
    
    # Version selector
    if rubric_history:
        version_options = [f"v{r.get('version', 1)}" for r in rubric_history]
        selected_version = st.selectbox(
            "Active Rubric Version:",
            options=version_options,
            index=active_idx if active_idx is not None else 0,
            key="rubric_version_selector"
        )
        if selected_version:
            new_idx = version_options.index(selected_version)
            if new_idx != active_idx:
                st.session_state.active_rubric_idx = new_idx
                # Update session state rubric
                active_rubric_dict, _, _ = get_active_rubric()
                rubric_list = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
                st.session_state.rubric = rubric_list
                # Reset editing criteria
                if active_rubric_dict:
                    st.session_state.editing_criteria = rubric_list.copy()
                st.rerun()
    
    # Display current rubric
    if st.session_state.editing_criteria:
        st.markdown("### Current Criteria")
        
        # Add criterion button
        if st.button("➕ Add Criterion", use_container_width=True):
            st.session_state.editing_criteria.append({
                "name": "",
                "description": "",
                "points": 0,
                "evidence": ""
            })
            st.rerun()
        
        for i, criterion in enumerate(st.session_state.editing_criteria):
            criterion_id = str(i + 1)
            
            # Simple layout with just the expander
            criterion_points = criterion.get('points', 0)
            points_emoji = ""
            if criterion_points > 0:
                points_emoji = "➕"
            elif criterion_points < 0:
                points_emoji = "➖"
            
            points_str = f" ({criterion_points:+d})" if criterion_points != 0 else " (0)"
            with st.expander(f"{points_emoji} {criterion.get('name', 'Unnamed Criterion')}", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    name_key = f"criterion_name_{i}_{st.session_state.active_rubric_idx if st.session_state.active_rubric_idx is not None else 0}"
                    name = st.text_input(
                        "Criterion Name",
                        value=criterion.get("name", ""),
                        key=name_key,
                        placeholder="e.g., Clarity and Conciseness"
                    )

                with col2:
                    points_key = f"criterion_points_{i}_{st.session_state.active_rubric_idx if st.session_state.active_rubric_idx is not None else 0}"
                    points = st.number_input(
                        "Points (-10 to 10)",
                        min_value=-10,
                        max_value=10,
                        value=int(criterion.get("points", 0)),
                        step=1,
                        key=points_key,
                        help="Positive values reward desired behaviors, negative values penalize undesired behaviors"
                    )

                desc_key = f"criterion_desc_{i}_{st.session_state.active_rubric_idx if st.session_state.active_rubric_idx is not None else 0}"
                description = st.text_area(
                    "Description",
                    value=criterion.get("description", ""),
                    key=desc_key,
                    placeholder="Describe what to look for in this criterion...",
                    height=100
                )

                # Update the criterion in the session state
                if name and description:
                    st.session_state.editing_criteria[i]["name"] = name
                    st.session_state.editing_criteria[i]["description"] = description
                    st.session_state.editing_criteria[i]["points"] = points
                
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.editing_criteria.pop(i)
                    st.rerun()
        
        # Update and Delete buttons in columns
        col_update, col_delete = st.columns([2, 1])
        
        with col_update:
            if st.button("💾 Update Rubric", use_container_width=True):
                if active_idx is not None and rubric_history:
                    # Create new version
                    new_version = rubric_history[active_idx].get("version", 1) + 1
                    updated_rubric_data = {
                        "version": new_version,
                        "rubric": st.session_state.editing_criteria.copy()
                    }
                    rubric_history.append(updated_rubric_data)
                    save_rubric_history(rubric_history)
                    st.session_state.active_rubric_idx = len(rubric_history) - 1
                    st.session_state.rubric = st.session_state.editing_criteria
                    
                    # Add a message to chat history about the rubric update
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"📋 Rubric manually updated to version {new_version}. Using this rubric for future responses."
                    })
                    
                    st.success(f"✓ Rubric updated to version {new_version}")
                    st.rerun()
                else:
                    rubric_list = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
                    st.session_state.rubric = rubric_list
        
        with col_delete:
            if st.button("🗑️ Delete Version", use_container_width=True, type="secondary"):
                if len(rubric_history) > 1:
                    # Delete the current version
                    deleted_version = rubric_history[active_idx].get("version", "?")
                    rubric_history.pop(active_idx)
                    
                    # Update active rubric index
                    if active_idx >= len(rubric_history):
                        st.session_state.active_rubric_idx = len(rubric_history) - 1
                    elif active_idx > 0:
                        st.session_state.active_rubric_idx = active_idx - 1
                    
                    # Save updated history
                    save_rubric_history(rubric_history)
                    
                    # Update current rubric
                    new_active_rubric_dict, new_active_idx, _ = get_active_rubric()
                    rubric_list = new_active_rubric_dict.get("rubric", []) if new_active_rubric_dict else []
                    st.session_state.rubric = rubric_list
                    st.session_state.active_rubric_idx = new_active_idx
                    
                    # Update editing criteria
                    if new_active_rubric_dict:
                        st.session_state.editing_criteria = rubric_list.copy()
                    
                    # Add system message
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"🗑️ Rubric version {deleted_version} deleted."
                    })
                    
                    st.success(f"✓ Rubric version {deleted_version} deleted")
                    st.rerun()
                else:
                    st.error("Cannot delete the last rubric version")
        
        # Fallback for when Update button condition is false
        if active_rubric_dict and active_idx is None:
            rubric_list = active_rubric_dict.get("rubric", [])
            st.session_state.rubric = rubric_list
    else:
        st.info("No rubric loaded. Use 'Infer Rubric' to create one from your conversation.")
    
    st.divider()

    # Analysis display
    with st.expander("🧠 Analysis", expanded=False):
        if st.session_state.current_analysis:
            # Escape HTML and use a scrollable container with custom CSS
            escaped_analysis = html.escape(st.session_state.current_analysis)
            st.markdown(
                f"""
                <div style='background-color: white; padding: 1rem; border-radius: 0.5rem; max-height: 300px; overflow-y: auto; border: 1px solid #e0e0e0;'>
                    <pre style='white-space: pre-wrap; font-family: system-ui, -apple-system, sans-serif; font-size: 0.875rem; color: black; margin: 0;'>{escaped_analysis}</pre>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Analysis will appear here when Claude responds with analysis tags.")

# These are duplicate sections that were moved into tab1 - keeping the sections inside tab1 only.

# Compare Rubrics Tab
with tab2:
    st.subheader("🔍 Compare Rubrics")
    st.markdown("Select two different rubric versions to see how they would affect the same writing task.")
    
    # Get rubric history
    _, _, rubric_history = get_active_rubric()
    
    if len(rubric_history) < 2:
        st.warning("You need at least 2 rubrics to compare. Create more rubric versions first.")
    else:
        rubric_options = [f"v{r.get('version', 1)}" for r in rubric_history]
        col1, col2 = st.columns(2)
        
        with col1:
            rubric_a_idx = st.selectbox("Rubric A:", options=list(range(len(rubric_history))), 
                                        format_func=lambda x: rubric_options[x], key="tab2_rubric_a_select")
        with col2:
            rubric_b_idx = st.selectbox("Rubric B:", options=list(range(len(rubric_history))), 
                                        format_func=lambda x: rubric_options[x], key="tab2_rubric_b_select")
        
        # Display selected rubrics
        st.markdown("---")
        col_rubric_a, col_rubric_b = st.columns(2)
        
        with col_rubric_a:
            rubric_a_version = rubric_history[rubric_a_idx].get('version', 1)
            st.markdown(f"### 📋 Rubric v{rubric_a_version}")
            # Pass rubric B as comparison to highlight differences in A
            display_rubric_criteria(rubric_history[rubric_a_idx], st, comparison_rubric_data=rubric_history[rubric_b_idx])

        with col_rubric_b:
            rubric_b_version = rubric_history[rubric_b_idx].get('version', 1)
            st.markdown(f"### 📋 Rubric v{rubric_b_version}")
            # Pass rubric A as comparison to highlight differences in B
            display_rubric_criteria(rubric_history[rubric_b_idx], st, comparison_rubric_data=rubric_history[rubric_a_idx])
        
        # Comparison task input
        with st.form("compare_form", clear_on_submit=False):
            compare_input = st.text_area("Writing task:", height=100,
                                        placeholder="Enter the writing task you want to compare with different rubrics...")
            compare_submit = st.form_submit_button("Generate Comparison")
        
        if compare_submit and compare_input.strip():
            if rubric_a_idx == rubric_b_idx:
                st.error("Please select different rubrics to compare.")
            else:
                # Generate comparison
                with st.spinner("Generating comparison..."):
                    result = compare_rubrics(compare_input, rubric_history[rubric_a_idx], rubric_history[rubric_b_idx])

                    # Store results in session state
                    st.session_state.rubric_comparison_results = {
                        "base_txt": result.get("base_txt", ""),
                        "a_txt": result.get("a_txt", ""),
                        "b_txt": result.get("b_txt", ""),
                        "key_diffs": result.get("key_diffs", ""),
                        "summary": result.get("summary", ""),
                        "rubric_a_idx": rubric_a_idx,
                        "rubric_b_idx": rubric_b_idx
                    }
                    st.rerun()

    # Display comparison results
    if st.session_state.rubric_comparison_results:
        results = st.session_state.rubric_comparison_results

        st.subheader("Comparison Results")

        # Key differences and summary at the top
        col_diff, col_summary = st.columns(2)

        with col_diff:
            st.markdown("### Key Differences")
            st.markdown(f"""
            <div style="
                height: 200px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {results["key_diffs"]}
            </div>
            """, unsafe_allow_html=True)

        with col_summary:
            st.markdown("### Summary")
            st.markdown(f"""
            <div style="
                height: 200px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {results["summary"]}
            </div>
            """, unsafe_allow_html=True)

        # Separator
        st.markdown("---")

        # Create three columns for side-by-side comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Base Draft")
            st.markdown("---")
            # Scrollable container for base draft
            st.markdown(f"""
            <div style="
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {results["base_txt"]}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Rubric A Revision")
            st.markdown("---")
            # Scrollable container for rubric A with diff highlighting
            diff_a = _md_diff_to_html_compare(results["a_txt"])
            st.markdown(f"""
            <div style="
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {diff_a}
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("### Rubric B Revision")
            st.markdown("---")
            # Scrollable container for rubric B with diff highlighting
            diff_b = _md_diff_to_html_compare(results["b_txt"])
            st.markdown(f"""
            <div style="
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {diff_b}
            </div>
            """, unsafe_allow_html=True)

# Build Rubric Tab
with tab3:
    st.subheader("📝 Build Rubric")
    st.markdown("Build a custom rubric by choosing between pairs of example texts. Your preferences will help refine the rubric criteria.")

    # Check if there's an active rubric - REQUIRED for this feature
    check_active_rubric_dict, _, _ = get_active_rubric()

    if not check_active_rubric_dict or not check_active_rubric_dict.get("rubric"):
        st.warning("⚠️ No active rubric found!")
        st.info("You need an active rubric to use the Build Rubric feature. Please go to the Chat tab and create or load a rubric first.")
        st.markdown("**How to get started:**")
        st.markdown("1. Go to the **💬 Chat** tab")
        st.markdown("2. Either start a conversation to create a new rubric, or load an existing rubric from history")
        st.markdown("3. Once you have an active rubric, come back here to refine it through preference-based comparisons")
    else:
        # Initialize session state for preference-based rubric building
        if 'pref_stage' not in st.session_state:
            st.session_state.pref_stage = 'setup'  # setup, comparing, complete
        if 'pref_writing_task' not in st.session_state:
            st.session_state.pref_writing_task = ""
        if 'pref_num_rounds' not in st.session_state:
            st.session_state.pref_num_rounds = 5
        if 'pref_current_round' not in st.session_state:
            st.session_state.pref_current_round = 0
        if 'pref_preferences' not in st.session_state:
            st.session_state.pref_preferences = []
        if 'pref_use_base_rubric' not in st.session_state:
            st.session_state.pref_use_base_rubric = False
        if 'pref_dimensions' not in st.session_state:
            st.session_state.pref_dimensions = ['clarity', 'structure', 'evidence', 'argumentation', 'style']
        if 'pref_text_pairs' not in st.session_state:
            st.session_state.pref_text_pairs = []
        if 'pref_refined_rubric' not in st.session_state:
            st.session_state.pref_refined_rubric = None

        # Stage 1: Initial Setup
        if st.session_state.pref_stage == 'setup':
            st.markdown("### 🎯 Setup Your Rubric Building Session")

            # Get criteria names from active rubric
            active_rubric_dict, _, _ = get_active_rubric()
            base_rubric_criteria = []
            if active_rubric_dict and active_rubric_dict.get("rubric"):
                # Extract criterion names from rubric
                base_rubric_criteria = [
                    criterion.get('name', 'Unnamed')
                    for criterion in active_rubric_dict.get("rubric", [])
                ]

            st.info(f"✨ Using criteria from your active rubric: {', '.join(base_rubric_criteria)}")

            with st.form("preference_setup_form"):
                # Writing task input
                writing_task = st.text_area(
                    "Writing Task Description:",
                    value=st.session_state.pref_writing_task,
                    height=100,
                    placeholder="E.g., 'Write a persuasive essay arguing for or against school uniforms'"
                )

                # Number of comparison rounds
                num_rounds = st.slider(
                    "Number of comparison rounds:",
                    min_value=3,
                    max_value=15,
                    value=st.session_state.pref_num_rounds,
                    help="More rounds = more refined rubric, but takes longer"
                )

                # Dimensions to focus on - always use criteria from active rubric (only one allowed)
                st.markdown("**Select ONE Criterion to Compare:**")
                selected_criterion = st.radio(
                    "Choose a criterion:",
                    options=base_rubric_criteria,
                    index=None,
                    key="selected_criterion_radio"
                )

                start_button = st.form_submit_button("🚀 Start Building Rubric")

            if start_button:
                if not writing_task.strip():
                    st.error("Please provide a writing task description.")
                elif selected_criterion is None:
                    st.error("Please select one criterion to evaluate.")
                else:
                    # Save setup configuration with single criterion
                    st.session_state.pref_writing_task = writing_task
                    st.session_state.pref_num_rounds = num_rounds
                    st.session_state.pref_use_base_rubric = True  # Always using base rubric
                    st.session_state.pref_dimensions = [selected_criterion]  # Single criterion in list
                    st.session_state.pref_current_round = 0
                    st.session_state.pref_preferences = []
                    st.session_state.pref_stage = 'comparing'
                    st.rerun()

        # Stage 2: Comparison Interface
        elif st.session_state.pref_stage == 'comparing':
            # Progress indicator
            progress = st.session_state.pref_current_round / st.session_state.pref_num_rounds
            st.progress(progress, text=f"Round {st.session_state.pref_current_round + 1} of {st.session_state.pref_num_rounds}")

            # Get current dimension for this round
            current_dimension = st.session_state.pref_dimensions[st.session_state.pref_current_round % len(st.session_state.pref_dimensions)]

            st.markdown(f"### 🔍 Comparing: **{current_dimension.title()}**")
            st.markdown(f"*Which text better demonstrates good {current_dimension}?*")

            # Check if we need to generate a new text pair
            if len(st.session_state.pref_text_pairs) <= st.session_state.pref_current_round:
                # Generate text pair using AI
                with st.spinner(f"Generating example texts for {current_dimension}..."):
                    # Get base rubric and extract criteria details
                    base_rubric = None
                    criteria_description = None
                    active_rubric_dict, _, _ = get_active_rubric()
                    if active_rubric_dict:
                        base_rubric = active_rubric_dict.get("rubric", None)
                        # Find the specific criterion that matches current_dimension
                        if base_rubric:
                            for criterion in base_rubric:
                                if criterion.get('name', '').lower() == current_dimension.lower() or criterion.get('name', '') == current_dimension:
                                    criteria_description = criterion.get('description', '')
                                    break

                    # Generate contrasting texts with explanations
                    text_a, text_b, explanation_a, explanation_b = generate_text_pair(
                        st.session_state.pref_writing_task,
                        current_dimension,
                        base_rubric,
                        criteria_description
                    )

                    st.session_state.pref_text_pairs.append({
                        'dimension': current_dimension,
                        'text_a': text_a,
                        'text_b': text_b,
                        'explanation_a': explanation_a,
                        'explanation_b': explanation_b,
                        'round': st.session_state.pref_current_round
                    })

            # Get current text pair
            current_pair = st.session_state.pref_text_pairs[st.session_state.pref_current_round]

            # Two-column layout for comparison
            col1, col2 = st.columns(2)

            with col1:
                text_a_html = simple_markdown_to_html(current_pair['text_a'])
                explanation_a = current_pair.get('explanation_a', '')
                explanation_a_html = simple_markdown_to_html(explanation_a) if explanation_a else ""

                st.markdown(f"""
                    <div style="border: 2px solid #2196F3; border-radius: 8px; padding: 16px; background-color: rgba(33, 150, 243, 0.05); min-height: 300px;">
                        <div style="font-size: 20px; font-weight: 600; color: #2196F3; margin-bottom: 12px;">📄 Text A</div>
                        <div style="line-height: 1.6; margin-bottom: 16px;">{text_a_html}</div>
                        {f'<div style="border-top: 1px solid rgba(33, 150, 243, 0.3); padding-top: 12px; margin-top: 12px;"><div style="font-size: 14px; font-weight: 600; color: #2196F3; margin-bottom: 6px;">Why this demonstrates the criterion:</div><div style="font-size: 14px; font-style: italic; color: #666;">{explanation_a_html}</div></div>' if explanation_a else ''}
                    </div>
                """, unsafe_allow_html=True)

                if st.button("✅ Choose Text A", key="choose_a", use_container_width=True, type="primary"):
                    # Record preference
                    preference = {
                        'round': st.session_state.pref_current_round,
                        'dimension': current_dimension,
                        'text_a': current_pair['text_a'],
                        'text_b': current_pair['text_b'],
                        'choice': 'a',
                        'timestamp': str(st.session_state.pref_current_round)
                    }
                    st.session_state.pref_preferences.append(preference)

                    # Move to next round
                    st.session_state.pref_current_round += 1

                    # Check if we're done
                    if st.session_state.pref_current_round >= st.session_state.pref_num_rounds:
                        st.session_state.pref_stage = 'complete'

                    st.rerun()

            with col2:
                text_b_html = simple_markdown_to_html(current_pair['text_b'])
                explanation_b = current_pair.get('explanation_b', '')
                explanation_b_html = simple_markdown_to_html(explanation_b) if explanation_b else ""

                st.markdown(f"""
                    <div style="border: 2px solid #FF9800; border-radius: 8px; padding: 16px; background-color: rgba(255, 152, 0, 0.05); min-height: 300px;">
                        <div style="font-size: 20px; font-weight: 600; color: #FF9800; margin-bottom: 12px;">📄 Text B</div>
                        <div style="line-height: 1.6; margin-bottom: 16px;">{text_b_html}</div>
                        {f'<div style="border-top: 1px solid rgba(255, 152, 0, 0.3); padding-top: 12px; margin-top: 12px;"><div style="font-size: 14px; font-weight: 600; color: #FF9800; margin-bottom: 6px;">Why this demonstrates the criterion:</div><div style="font-size: 14px; font-style: italic; color: #666;">{explanation_b_html}</div></div>' if explanation_b else ''}
                    </div>
                """, unsafe_allow_html=True)

                if st.button("✅ Choose Text B", key="choose_b", use_container_width=True, type="primary"):
                    # Record preference
                    preference = {
                        'round': st.session_state.pref_current_round,
                        'dimension': current_dimension,
                        'text_a': current_pair['text_a'],
                        'text_b': current_pair['text_b'],
                        'choice': 'b',
                        'timestamp': str(st.session_state.pref_current_round)
                    }
                    st.session_state.pref_preferences.append(preference)

                    # Move to next round
                    st.session_state.pref_current_round += 1

                    # Check if we're done
                    if st.session_state.pref_current_round >= st.session_state.pref_num_rounds:
                        st.session_state.pref_stage = 'complete'

                    st.rerun()

                # Optional: Add reasoning text box
                # with st.expander("💭 Add your reasoning (optional)"):
                #     reasoning = st.text_area(
                #         "Why did you prefer this text?",
                #         key=f"reasoning_{st.session_state.pref_current_round}",
                #         placeholder="E.g., 'Text A has clearer topic sentences and better transitions...'"
                #     )

                # # Show current preferences in sidebar
                # with st.sidebar:
                #     st.markdown("### 📊 Your Preferences So Far")
                #     if st.session_state.pref_preferences:
                #         for i, pref in enumerate(st.session_state.pref_preferences):
                #             st.markdown(f"**Round {i+1}:** {pref['dimension'].title()} → Text {pref['choice'].upper()}")
                #     else:
                #         st.info("No preferences recorded yet")

                # Reset button
                if st.button("🔄 Start Over", key="reset_comparison"):
                    st.session_state.pref_stage = 'setup'
                    st.session_state.pref_current_round = 0
                    st.session_state.pref_preferences = []
                    st.session_state.pref_text_pairs = []
                    st.rerun()

        # Stage 3: Complete - Show final rubric
        elif st.session_state.pref_stage == 'complete':
            st.markdown("### ✅ Rubric Building Complete!")
            st.success(f"You completed {len(st.session_state.pref_preferences)} comparisons. Now generating your custom rubric...")

            # Generate rubric if not already done
            if st.session_state.pref_refined_rubric is None:
                with st.spinner("Analyzing your preferences and creating rubric..."):
                    try:
                        # Get base rubric and extract criterion info
                        base_rubric = None
                        criteria_info = {}
                        import copy

                        if st.session_state.pref_use_base_rubric:
                            active_rubric_dict, _, _ = get_active_rubric()
                            if active_rubric_dict:
                                base_rubric = active_rubric_dict.get("rubric", None)

                                # Extract full criterion information for the selected dimension
                                selected_criterion_name = st.session_state.pref_dimensions[0]  # Only one criterion now
                                if base_rubric:
                                    for criterion in base_rubric:
                                        if criterion.get('name', '').lower() == selected_criterion_name.lower():
                                            criteria_info = {
                                                'name': criterion.get('name', ''),
                                                'category': criterion.get('category', ''),
                                                'description': criterion.get('description', ''),
                                                'points': criterion.get('points', 0)
                                            }
                                            # IMPORTANT: Save a deep copy of the original criterion for comparison
                                            st.session_state.pref_original_criterion = copy.deepcopy(criterion)
                                            break

                        # Generate refined rubric with full criterion info
                        rubric_data = analyze_preferences_and_generate_rubric(
                            st.session_state.pref_preferences,
                            st.session_state.pref_writing_task,
                            criteria_info,
                            base_rubric
                        )

                        st.session_state.pref_refined_rubric = rubric_data
                    except Exception as e:
                        st.error(f"Error generating rubric: {str(e)}")
                        st.session_state.pref_refined_rubric = False  # Mark as failed, not None

            # Display the refined criterion
            if st.session_state.pref_refined_rubric and st.session_state.pref_refined_rubric is not False:
                st.markdown("### 📋 Refined Criterion Based on Your Preferences")

                # The refined criterion is a single object, not wrapped in a rubric array
                refined_criterion = st.session_state.pref_refined_rubric

                # Get the original criterion from session state (saved before any updates)
                # This ensures we show the ORIGINAL criterion, not the updated one
                original_criterion = st.session_state.get('pref_original_criterion', None)

                # Display comparison
                st.markdown("#### Before vs. After Comparison")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### 📌 Original Criterion")
                    if original_criterion:
                        st.markdown(f"**{original_criterion.get('name', 'N/A')}**")
                        st.markdown(f"*Category:* {original_criterion.get('category', 'N/A')}")
                        orig_points = original_criterion.get('points', 0)
                        st.markdown(f"*Points:* {orig_points:+d}" if orig_points != 0 else f"*Points:* 0")
                        st.markdown(f"*Description:*")
                        st.info(original_criterion.get('description', 'N/A'))
                        if original_criterion.get('evidence'):
                            with st.expander("📌 Original Evidence"):
                                evidence = original_criterion.get('evidence', '')
                                # Check if evidence is a string or list
                                if isinstance(evidence, str):
                                    st.markdown(evidence)
                                else:
                                    for ev in evidence:
                                        st.markdown(f"- {ev}")
                    else:
                        st.info("No original criterion found")

                with col2:
                    st.markdown("##### ✨ Refined Criterion")
                    st.markdown(f"**{refined_criterion.get('name', 'N/A')}**")
                    st.markdown(f"*Category:* {refined_criterion.get('category', 'N/A')}")
                    refined_points = refined_criterion.get('points', 0)
                    st.markdown(f"*Points:* {refined_points:+d}" if refined_points != 0 else f"*Points:* 0")
                    st.markdown(f"*Description:*")
                    st.success(refined_criterion.get('description', 'N/A'))
                    if refined_criterion.get('evidence'):
                        with st.expander("✨ Refined Evidence"):
                            evidence = refined_criterion.get('evidence', [])
                            # Check if evidence is a string or list
                            if isinstance(evidence, str):
                                st.markdown(evidence)
                            else:
                                for ev in evidence:
                                    st.markdown(f"- {ev}")

                # Show preference summary
                with st.expander("📊 View Your Preferences"):
                    st.markdown("#### Comparison Summary")
                    for i, pref in enumerate(st.session_state.pref_preferences):
                        st.markdown(f"**Round {i+1}: {pref['dimension'].title()}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            chosen_marker = "✅" if pref['choice'] == 'a' else ""
                            st.markdown(f"{chosen_marker} **Text A**")
                            st.text(pref['text_a'][:150] + "...")
                        with col2:
                            chosen_marker = "✅" if pref['choice'] == 'b' else ""
                            st.markdown(f"{chosen_marker} **Text B**")
                            st.text(pref['text_b'][:150] + "...")
                        st.markdown("---")

                # Action buttons
                st.markdown("---")
                st.markdown("#### Do you want to keep this refined criterion?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Yes, Update My Rubric", key="accept_criterion", use_container_width=True):
                        # Get current active rubric
                        active_rubric_dict, _, rubric_history = get_active_rubric()

                        if active_rubric_dict and active_rubric_dict.get("rubric"):
                            # Create a deep copy of the active rubric
                            import copy
                            updated_rubric = copy.deepcopy(active_rubric_dict.get("rubric", []))

                            # Find and update the specific criterion
                            selected_criterion_name = st.session_state.pref_dimensions[0]
                            # refined_criterion is already available from the display section above

                            criterion_updated = False
                            for i, crit in enumerate(updated_rubric):
                                if crit.get('name', '').lower() == selected_criterion_name.lower():
                                    # Update this criterion with the refined version
                                    updated_rubric[i] = refined_criterion
                                    criterion_updated = True
                                    break

                            if not criterion_updated:
                                st.error(f"Could not find criterion '{selected_criterion_name}' in active rubric.")
                            else:
                                # Create new rubric version
                                version = next_version_number()
                                
                                new_entry = {
                                    "version": version,
                                    "rubric": updated_rubric,
                                    "is_active": True
                                    # "source": "preference_builder_update"
                                }

                                # Mark all others as inactive
                                for entry in rubric_history:
                                    entry['is_active'] = False

                                # Add new entry at the end of the list
                                rubric_history.append(new_entry)
                                save_rubric_history(rubric_history)

                                # Update session state - set active index to the last entry
                                st.session_state.rubric = updated_rubric
                                st.session_state.active_rubric_idx = len(rubric_history) - 1
                                # IMPORTANT: Also update editing_criteria so sidebar shows updated rubric
                                st.session_state.editing_criteria = copy.deepcopy(updated_rubric)

                                st.success(f"✅ Criterion '{selected_criterion_name}' has been updated in your active rubric (v{version})!")
                                st.balloons()
                                st.rerun()
                        else:
                            st.error("Could not find active rubric to update.")

                with col2:
                    if st.button("❌ No, Discard Changes", key="reject_criterion", use_container_width=True):
                        # Reset to setup stage without saving
                        st.session_state.pref_stage = 'setup'
                        st.session_state.pref_current_round = 0
                        st.session_state.pref_preferences = []
                        st.session_state.pref_text_pairs = []
                        st.session_state.pref_refined_rubric = None
                        if 'pref_original_criterion' in st.session_state:
                            del st.session_state.pref_original_criterion
                        st.info("Changes discarded. You can start a new refinement session.")
                        st.rerun()

            elif st.session_state.pref_refined_rubric is False:
                st.error("Failed to generate rubric. Please try again.")
                if st.button("🔄 Try Again"):
                    st.session_state.pref_refined_rubric = None
                    st.rerun()

