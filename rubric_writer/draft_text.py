"""Draft parsing, sentence helpers, markdown diff utilities."""
from rubric_writer.imports import *

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

def parse_draft_content(content: str):
    """
    Parse content to extract draft sections wrapped in <draft></draft> tags.
    Returns a list of tuples: (text_before, draft_content, text_after) for each draft found,
    or None if no drafts are found.
    """
    pattern = r'<draft>(.*?)</draft>'
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if not matches:
        return None

    parts = []
    last_end = 0

    for match in matches:
        text_before = content[last_end:match.start()]
        draft_content = match.group(1).strip()
        parts.append({
            'before': text_before,
            'draft': draft_content,
            'start': match.start(),
            'end': match.end()
        })
        last_end = match.end()

    # Add any remaining text after the last draft
    if last_end < len(content):
        parts.append({
            'after': content[last_end:]
        })

    return parts


def split_draft_into_sentences(draft_text: str) -> list:
    """Split draft text into sentences, preserving paragraph structure."""
    paragraphs = draft_text.split('\n\n')
    sentences = []
    for para in paragraphs:
        if not para.strip():
            continue
        # Split on sentence-ending punctuation followed by space + uppercase letter
        para_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para.strip())
        sentences.extend([s.strip() for s in para_sentences if s.strip()])
    return sentences


def replace_sentences_in_draft(draft: str, replacements: list) -> str:
    """Replace each sentence in draft with its corresponding replacement in-place.

    replacements: list of {"original": str, "replacement": str} dicts.
    Each original sentence is found in the draft and swapped for its replacement,
    preserving all surrounding text.
    """
    result = draft
    for item in replacements:
        original = item.get("original", "")
        replacement = item.get("replacement", "")
        if original and original in result:
            result = result.replace(original, replacement, 1)
    return result


def strip_draft_tags_for_streaming(content: str) -> str:
    """
    Strip <draft> tags from content for display during streaming.
    Replaces <draft>content</draft> with just the content inside the tags.
    This ensures streaming display matches what's shown in editable text areas after rerun.
    """
    # Replace complete <draft>...</draft> with just the inner content
    pattern = r'<draft>(.*?)</draft>'
    result = re.sub(pattern, r'\1', content, flags=re.DOTALL)

    # Handle incomplete/partial draft tags during streaming
    # If there's an opening <draft> without closing, show content up to the tag
    if '<draft>' in result and '</draft>' not in result:
        # Keep partial draft content visible during streaming
        result = result.replace('<draft>', '')

    return result
