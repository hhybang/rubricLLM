"""Parse assistant output, streaming, feedback formatting."""
from rubric_writer.imports import *
from rubric_writer.draft_text import strip_draft_tags_for_streaming

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
    Returns: dict with evaluation details and JSON summary, or None if not found
    """
    result = {
        'evaluation_text': None,
        'json_summary': None,
        'criteria_details': {}
    }

    # First, try to extract <evaluation> tags content
    eval_pattern = r'<evaluation>(.*?)</evaluation>'
    eval_match = re.search(eval_pattern, full_text, re.DOTALL)

    if eval_match:
        result['evaluation_text'] = eval_match.group(1).strip()

        # Parse individual criterion sections from evaluation text
        # Find all ### headers and extract sections
        # Use finditer to get all matches with their positions
        criterion_pattern = r'### (.+?)(?=\n###|\Z)'
        matches = re.finditer(criterion_pattern, result['evaluation_text'], re.DOTALL)

        for match in matches:
            section = match.group(0)  # Get the full matched text including ###
            lines = section.split('\n')
            # Remove ### prefix from the first line
            criterion_name = lines[0].replace('###', '').strip()

            # Initialize criterion details
            criterion_data = {
                'priority': None,
                'achievement_level': None,
                'evidence': [],
                'rationale': None,
                'to_reach_next_level': None
            }

            current_field = None
            content_buffer = []

            for line in lines[1:]:
                line = line.strip()

                # Handle both old format (Weight) and new format (Priority)
                if line.startswith('**Weight**:'):
                    criterion_data['weight'] = line.replace('**Weight**:', '').strip()
                elif line.startswith('**Priority**:'):
                    priority_text = line.replace('**Priority**:', '').strip()
                    # Extract number from "#N" format
                    priority_match = re.search(r'#?(\d+)', priority_text)
                    if priority_match:
                        criterion_data['priority'] = int(priority_match.group(1))
                elif line.startswith('**Achievement Level**:'):
                    level_text = line.replace('**Achievement Level**:', '').strip()
                    criterion_data['achievement_level'] = level_text
                elif line.startswith('**Evidence from draft**:'):
                    current_field = 'evidence'
                    content_buffer = []
                elif line.startswith('**Rationale**:'):
                    if current_field == 'evidence':
                        criterion_data['evidence'] = content_buffer
                    current_field = 'rationale'
                    content_buffer = []
                elif line.startswith('**To reach next level**:'):
                    if current_field == 'rationale':
                        criterion_data['rationale'] = ' '.join(content_buffer)
                    current_field = 'to_reach_next_level'
                    content_buffer = []
                elif line.startswith('---'):
                    # End of criterion section
                    if current_field == 'to_reach_next_level':
                        criterion_data['to_reach_next_level'] = ' '.join(content_buffer)
                    break
                elif current_field:
                    if line.startswith('- '):
                        content_buffer.append(line[2:])  # Remove bullet
                    elif line:
                        content_buffer.append(line)

            # Save any remaining buffer
            if current_field == 'rationale':
                criterion_data['rationale'] = ' '.join(content_buffer)
            elif current_field == 'to_reach_next_level':
                criterion_data['to_reach_next_level'] = ' '.join(content_buffer)

            result['criteria_details'][criterion_name] = criterion_data

    # Extract JSON summary (look for it anywhere in the text, not just in tags)
    json_pattern = r'```json\s*(.*?)\s*```'
    json_match = re.search(json_pattern, full_text, re.DOTALL)

    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            result['json_summary'] = json_data
        except json.JSONDecodeError:
            pass

    # Return None if we found nothing useful
    if not result['evaluation_text'] and not result['json_summary']:
        return None

    return result

def display_rubric_assessment(assessment_data, message_id=None, draft_text=None):
    """Display rubric assessment in a card-based layout with dimension checklists and evidence highlights"""
    if not assessment_data:
        return

    st.markdown("---")

    # Initialize feedback storage in session state if needed
    if 'assessment_feedback' not in st.session_state:
        st.session_state.assessment_feedback = {}

    # Use message_id for unique key, fallback to id(assessment_data)
    assessment_key = f"assessment_{message_id}" if message_id else f"assessment_{id(assessment_data)}"
    if assessment_key not in st.session_state.assessment_feedback:
        st.session_state.assessment_feedback[assessment_key] = {}

    # Get assessment summary from JSON
    overall_assessment = None
    criteria_scores = []
    evidence_highlights = []

    if assessment_data.get('json_summary'):
        json_data = assessment_data['json_summary']
        overall_assessment = json_data.get('overall_assessment')
        criteria_scores = json_data.get('criteria_scores', [])
        evidence_highlights = json_data.get('evidence_highlights', [])

    # Helper function to get level info (star-based system)
    def get_level_info(achievement_level):
        if not achievement_level:
            return "#999", "❓"
        level_lower = achievement_level.lower()
        if 'excellent' in level_lower:
            return "#4CAF50", "⭐⭐⭐"
        elif 'good' in level_lower:
            return "#2196F3", "⭐⭐"
        elif 'fair' in level_lower:
            return "#FF9800", "⭐"
        elif 'needs work' in level_lower or 'needs_work' in level_lower:
            return "#E65100", "◇"
        else:  # weak
            return "#F44336", "☆"

    # Display assessment header
    with st.expander("📊 Rubric Assessment", expanded=False):
        # Show thinking if available
        if assessment_data.get('thinking'):
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(assessment_data['thinking'])

        # Sort criteria by priority
        sorted_criteria = sorted(criteria_scores, key=lambda x: x.get('priority', 99))

        # --- Summary dashboard ---
        if sorted_criteria:
            # Count levels
            level_counts = {"excellent": 0, "good": 0, "fair": 0, "needs work": 0, "weak": 0}
            total_dims_met = 0
            total_dims_all = 0
            for crit in sorted_criteria:
                lvl = crit.get('achievement_level', '').lower()
                for k in level_counts:
                    if k in lvl:
                        level_counts[k] += 1
                        break
                total_dims_met += crit.get('dimensions_met', 0)
                total_dims_all += crit.get('dimensions_total', 0)
            overall_pct = round(total_dims_met / total_dims_all * 100) if total_dims_all > 0 else 0

            # Overall score bar — compute 1-5 rating and label from percentage
            if overall_pct >= 90:
                overall_rating, overall_label, bar_color = 5, "Excellent", "#4CAF50"
            elif overall_pct >= 75:
                overall_rating, overall_label, bar_color = 4, "Good", "#2196F3"
            elif overall_pct >= 50:
                overall_rating, overall_label, bar_color = 3, "Fair", "#FF9800"
            elif overall_pct >= 25:
                overall_rating, overall_label, bar_color = 2, "Needs Work", "#E65100"
            else:
                overall_rating, overall_label, bar_color = 1, "Weak", "#F44336"
            summary_html = f'''<div style="background:linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);border:1px solid #e0e0e0;border-radius:12px;padding:20px;margin-bottom:16px;">
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
  <div>
    <span style="font-size:1.3em;font-weight:700;">Overall Score</span>
    <span style="display:inline-block;margin-left:10px;padding:3px 12px;background:{bar_color}20;color:{bar_color};border-radius:14px;font-size:0.95em;font-weight:700;">{overall_label} ({overall_rating})</span>
  </div>
  <span style="font-size:2em;font-weight:800;color:{bar_color};">{overall_pct}%</span>
</div>
<div style="background:#e9ecef;border-radius:8px;height:12px;overflow:hidden;margin-bottom:14px;">
  <div style="background:{bar_color};height:100%;width:{overall_pct}%;border-radius:8px;transition:width 0.5s;"></div>
</div>
<div style="font-size:0.85em;color:#666;margin-bottom:10px;">{total_dims_met} of {total_dims_all} dimensions met across {len(sorted_criteria)} criteria</div>
<div style="display:flex;gap:6px;flex-wrap:wrap;">'''
            level_configs = [
                ("excellent", "Excellent", "#E8F5E9", "#2E7D32"),
                ("good", "Good", "#E3F2FD", "#1565C0"),
                ("fair", "Fair", "#FFF3E0", "#E65100"),
                ("needs work", "Needs Work", "#FBE9E7", "#BF360C"),
                ("weak", "Weak", "#FFEBEE", "#C62828"),
            ]
            for key, label, bg, fg in level_configs:
                count = level_counts.get(key, 0)
                if count > 0:
                    summary_html += f'<span style="padding:4px 12px;background:{bg};color:{fg};border-radius:16px;font-size:0.85em;font-weight:600;">{count} {label}</span>'
            summary_html += '</div></div>'
            st.markdown(summary_html, unsafe_allow_html=True)

        # Overall assessment narrative
        if overall_assessment:
            st.markdown(f"*{overall_assessment}*")
            st.markdown("")

        # Build a lookup from (criterion, dimension_id) to evidence_highlights quotes
        highlight_lookup = {}
        for ev in evidence_highlights:
            key = (ev.get('criterion', ''), ev.get('dimension_id', ''))
            if key not in highlight_lookup:
                highlight_lookup[key] = []
            highlight_lookup[key].append({
                'quote': ev.get('quote', ''),
                'relevance': ev.get('relevance', '')
            })

        # --- Criteria cards ---
        for crit in sorted_criteria:
            crit_name = crit.get('name', 'Unknown')
            priority = crit.get('priority', 'N/A')
            achievement_level = crit.get('achievement_level', 'N/A')
            dimensions_detail = crit.get('dimensions_detail', [])
            dims_met = crit.get('dimensions_met', 0)
            dims_total = crit.get('dimensions_total', 0)
            improvement_explanation = crit.get('improvement_explanation', '')

            level_color, level_emoji = get_level_info(achievement_level)
            pct = round(dims_met / dims_total * 100) if dims_total > 0 else 0
            # Compute 1-5 rating for this criterion
            crit_pct_frac = dims_met / dims_total if dims_total > 0 else 0
            if crit_pct_frac >= 0.90: crit_rating = 5
            elif crit_pct_frac >= 0.75: crit_rating = 4
            elif crit_pct_frac >= 0.50: crit_rating = 3
            elif crit_pct_frac >= 0.25: crit_rating = 2
            else: crit_rating = 1

            # Criterion card header with progress bar
            card_header = f'''<div style="background:white;border:1px solid #e0e0e0;border-left:4px solid {level_color};border-radius:8px;padding:14px 16px;margin-bottom:4px;">
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
  <div>
    <span style="font-size:0.75em;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Priority #{priority}</span>
    <div style="font-size:1.05em;font-weight:600;margin-top:2px;">{level_emoji} {crit_name}</div>
  </div>
  <div style="text-align:right;">
    <span style="background:{level_color}20;color:{level_color};padding:3px 10px;border-radius:12px;font-size:0.8em;font-weight:600;">{achievement_level} ({crit_rating})</span>
    <div style="font-size:0.8em;color:#888;margin-top:4px;">{dims_met}/{dims_total} dimensions · {pct}%</div>
  </div>
</div>
<div style="background:#e9ecef;border-radius:6px;height:6px;overflow:hidden;margin-top:10px;">
  <div style="background:{level_color};height:100%;width:{pct}%;border-radius:6px;"></div>
</div>
</div>'''
            st.markdown(card_header, unsafe_allow_html=True)

            with st.expander(f"View details — {crit_name}", expanded=False):
                # Dimension checklist as styled items
                for dim in dimensions_detail:
                    dim_id = dim.get('id', '')
                    dim_label = dim.get('label', 'Unknown dimension')
                    dim_met = dim.get('met', False)
                    dim_evidence = dim.get('evidence', '')
                    linked_highlights = highlight_lookup.get((crit_name, dim_id), [])

                    if dim_met:
                        st.markdown(f'''<div style="display:flex;align-items:flex-start;gap:8px;padding:8px 12px;background:#f0faf0;border-radius:6px;margin-bottom:4px;border:1px solid #c8e6c9;">
<span style="color:#2E7D32;font-size:1.1em;flex-shrink:0;">✅</span>
<div><span style="font-weight:500;">{dim_label}</span></div>
</div>''', unsafe_allow_html=True)
                        evidence_text = ""
                        if linked_highlights:
                            for hl in linked_highlights:
                                quote = hl.get('quote', '')
                                if quote:
                                    evidence_text = quote
                                    break
                        if not evidence_text and dim_evidence:
                            evidence_text = dim_evidence
                        if evidence_text:
                            st.markdown(f'''<div style="margin-left:32px;padding:6px 12px;border-left:3px solid #a5d6a7;color:#555;font-size:0.88em;font-style:italic;margin-bottom:6px;">"{evidence_text}"</div>''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''<div style="display:flex;align-items:flex-start;gap:8px;padding:8px 12px;background:#fef0ef;border-radius:6px;margin-bottom:4px;border:1px solid #ffcdd2;">
<span style="color:#C62828;font-size:1.1em;flex-shrink:0;">❌</span>
<div><span style="font-weight:500;">{dim_label}</span></div>
</div>''', unsafe_allow_html=True)
                        evidence_text = ""
                        if linked_highlights:
                            for hl in linked_highlights:
                                quote = hl.get('quote', '')
                                if quote:
                                    evidence_text = quote
                                    break
                        if not evidence_text and dim_evidence:
                            evidence_text = dim_evidence
                        if evidence_text:
                            st.markdown(f'''<div style="margin-left:32px;padding:6px 12px;border-left:3px solid #ef9a9a;color:#555;font-size:0.88em;font-style:italic;margin-bottom:6px;">"{evidence_text}"</div>''', unsafe_allow_html=True)

                # Improvement explanation
                if improvement_explanation and 'excellent' not in achievement_level.lower():
                    st.markdown(f'''<div style="background:#FFF8E1;border:1px solid #FFE082;border-radius:8px;padding:12px 14px;margin-top:8px;">
<div style="font-weight:600;font-size:0.9em;color:#F57F17;margin-bottom:4px;">💡 To improve</div>
<div style="font-size:0.9em;color:#555;">{improvement_explanation}</div>
</div>''', unsafe_allow_html=True)

        # Evidence Highlights Section
        if evidence_highlights and draft_text:
            st.markdown("---")
            st.markdown("### 📄 Draft with Evidence Highlights")
            st.markdown("*Hover over highlighted text to see the criterion and relevance explanation.*")

            # Build evidence map with actual positions
            all_evidence = []
            for ev in evidence_highlights:
                quote = ev.get("quote", "").strip()
                if quote:
                    # Find actual position by searching for the quote
                    start_idx = draft_text.find(quote)
                    if start_idx == -1:
                        # Try finding first 30 chars as a fallback
                        if len(quote) > 30:
                            start_idx = draft_text.find(quote[:30])

                    if start_idx != -1:
                        all_evidence.append({
                            "criterion": ev.get("criterion", ""),
                            "quote": quote,
                            "start_index": start_idx,
                            "end_index": start_idx + len(quote),
                            "relevance": ev.get("relevance", ""),
                            "dimension_id": ev.get("dimension_id", ""),
                            "dimension_met": ev.get("dimension_met", True)
                        })

            if all_evidence:
                # Sort by start_index
                sorted_evidence = sorted(all_evidence, key=lambda x: x.get("start_index", 0))

                # Remove overlapping highlights
                non_overlapping = []
                last_end = 0
                for ev in sorted_evidence:
                    if ev["start_index"] >= last_end:
                        non_overlapping.append(ev)
                        last_end = ev["end_index"]
                sorted_evidence = non_overlapping

                # Define colors for different criteria
                criterion_colors = {}
                colors = ["#ffeb3b", "#81d4fa", "#a5d6a7", "#ffcc80", "#ce93d8", "#ef9a9a", "#80cbc4"]
                for i, crit in enumerate(sorted_criteria):
                    criterion_colors[crit.get("name", f"Criterion {i+1}")] = colors[i % len(colors)]

                # Build dimension label lookup: (criterion_name, dimension_id) -> dimension_label
                dimension_labels = {}
                for crit in sorted_criteria:
                    crit_name = crit.get("name", "")
                    for dim in crit.get("dimensions_detail", []):
                        dim_id = dim.get("id", "")
                        dim_label = dim.get("label", "")
                        if dim_id:
                            dimension_labels[(crit_name, dim_id)] = dim_label

                # Legend showing criterion colors with colored boxes
                st.markdown("**Legend:**")
                legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">'
                for crit in sorted_criteria:
                    crit_name = crit.get("name", "")
                    color = criterion_colors.get(crit_name, "#ffeb3b")
                    legend_html += f'<span style="background-color: {color}; padding: 2px 8px; border-radius: 3px; font-size: 0.9em;">{crit_name}</span>'
                legend_html += '</div>'
                st.markdown(legend_html, unsafe_allow_html=True)

                # Build inline highlighted draft with tooltips (same as Evaluate: Alignment tab)
                tooltip_css = """<style>
.evidence-highlight { position: relative; cursor: help; padding: 1px 2px; border-radius: 2px; }
.evidence-highlight .tooltip-text { visibility: hidden; background-color: #333; color: #fff; text-align: left; padding: 8px 12px; border-radius: 6px; position: absolute; z-index: 1000; bottom: 125%; left: 50%; transform: translateX(-50%); width: 280px; font-size: 0.85em; line-height: 1.4; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
.evidence-highlight .tooltip-text::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #333 transparent transparent transparent; }
.evidence-highlight:hover .tooltip-text { visibility: visible; }
.tooltip-criterion { font-weight: bold; color: #ffc107; margin-bottom: 5px; }
</style>"""

                highlighted_html = tooltip_css
                highlighted_html += '<div style="line-height: 1.8; padding: 10px; background-color: #f9f9f9; border-radius: 8px; border: 1px solid #ddd;">'

                last_end = 0
                for ev in sorted_evidence:
                    start = ev.get("start_index", 0)
                    end = ev.get("end_index", 0)
                    criterion = ev.get("criterion", "")
                    dimension_id = ev.get("dimension_id", "")
                    dimension_met = ev.get("dimension_met", True)
                    relevance = ev.get("relevance", "")
                    color = criterion_colors.get(criterion, "#ffeb3b")

                    # Look up dimension label
                    dimension_label = dimension_labels.get((criterion, dimension_id), dimension_id)

                    # Build status indicator
                    status_icon = "✅" if dimension_met else "❌"
                    status_color = "#4CAF50" if dimension_met else "#F44336"

                    if start >= last_end and end > start and start < len(draft_text):
                        # Add unhighlighted text before this highlight
                        highlighted_html += draft_text[last_end:start].replace("\n", "<br>")

                        # Add highlighted text with tooltip showing criterion + dimension + status
                        quote_text = draft_text[start:end].replace("\n", "<br>")
                        tooltip_content = f'<div class="tooltip-criterion">{criterion}</div>'
                        if dimension_label:
                            tooltip_content += f'<div style="color: {status_color}; margin-bottom: 5px;">{status_icon} {dimension_label}</div>'
                        if relevance:
                            tooltip_content += f'<div style="font-size: 0.9em; opacity: 0.9;">{relevance}</div>'

                        highlighted_html += f'<span class="evidence-highlight" style="background-color: {color};">{quote_text}<span class="tooltip-text">{tooltip_content}</span></span>'

                        last_end = end

                # Add remaining text
                if last_end < len(draft_text):
                    highlighted_html += draft_text[last_end:].replace("\n", "<br>")

                highlighted_html += '</div>'

                st.markdown(highlighted_html, unsafe_allow_html=True)
            else:
                # No evidence found in draft text - show without highlights
                st.info("No evidence quotes could be matched to the draft text.")
                st.markdown(draft_text)
        elif draft_text and not evidence_highlights:
            # Show draft without highlights if no evidence_highlights provided
            st.markdown("---")
            st.markdown("### 📄 Draft")
            st.markdown(draft_text)



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

def stream_without_analysis(stream, response_placeholder=None, message_id=None, thinking_placeholder=None):
    """Stream response while hiding analysis and rubric_assessment tags.

    Note: This function strips out any rubric_assessment tags the model may generate
    during normal chat. Assessments should only be displayed when explicitly requested
    via the 'Assess Draft' button.

    With extended thinking enabled, this also captures thinking content but does NOT
    stream it - thinking is buffered silently and only shown in a collapsed expander
    after the response is complete.

    Both `response_placeholder` and `thinking_placeholder` are optional. The live
    UI passes Streamlit `st.empty()` widgets so streaming chunks render in place;
    the synthetic-study pipeline (which has no UI) passes None and just consumes
    the final return value. Behavior is otherwise identical: same tag stripping,
    same final-content extraction. The only difference is whether intermediate
    chunks are displayed.
    """
    full_response = ""
    thinking_content = ""
    started_main_content = False

    # Show a simple thinking indicator while thinking (no streaming of thinking content).
    # Use a small CSS spinner so the icon visibly rotates while waiting,
    # matching Streamlit's native loading affordance.
    if thinking_placeholder is not None:
        thinking_placeholder.markdown(
            """
<div style="display:flex;align-items:center;gap:8px;color:#666;font-style:italic;">
  <div style="width:14px;height:14px;border:2px solid #d0d0d0;border-top-color:#888;border-radius:50%;animation:thinking-spin 0.8s linear infinite;"></div>
  <span>Thinking...</span>
</div>
<style>@keyframes thinking-spin{to{transform:rotate(360deg);}}</style>
""",
            unsafe_allow_html=True,
        )

    # Handle streaming with extended thinking
    for event in stream:
        # Handle thinking events
        if hasattr(event, 'type'):
            if event.type == 'content_block_start':
                if hasattr(event, 'content_block') and event.content_block.type == 'text':
                    # Clear thinking indicator when main content starts
                    if thinking_placeholder is not None and not started_main_content:
                        thinking_placeholder.empty()
                        started_main_content = True
            elif event.type == 'content_block_delta':
                if hasattr(event, 'delta'):
                    if event.delta.type == 'thinking_delta':
                        # Buffer thinking content silently - don't stream it
                        thinking_content += event.delta.thinking
                    elif event.delta.type == 'text_delta':
                        full_response += event.delta.text

                        # Stop streaming if we hit rubric_assessment tag (strip it out)
                        if '<rubric_assessment>' in full_response:
                            # Get content before rubric_assessment tag
                            content_before_assessment = full_response.split('<rubric_assessment>')[0]
                            _, main_content = parse_analysis_and_content(content_before_assessment)
                            # Strip draft tags for streaming display to match post-rerun appearance
                            if response_placeholder is not None:
                                streaming_content = strip_draft_tags_for_streaming(main_content)
                                response_placeholder.markdown(streaming_content)
                            continue

                        # Parse the current accumulated response to filter out analysis
                        if response_placeholder is not None:
                            _, main_content = parse_analysis_and_content(full_response)
                            # Strip draft tags for streaming display to match post-rerun appearance
                            streaming_content = strip_draft_tags_for_streaming(main_content)
                            response_placeholder.markdown(streaming_content + "▌")

    # Final parse to ensure clean output
    analysis_content, main_content = parse_analysis_and_content(full_response)

    # Strip any rubric_assessment content from main_content
    # (model should not be generating assessments during normal chat)
    if '<rubric_assessment>' in main_content:
        main_content = main_content.split('<rubric_assessment>')[0].strip()

    # Display the final content (without assessment)
    # Strip draft tags for streaming display to match post-rerun appearance
    if response_placeholder is not None:
        streaming_content = strip_draft_tags_for_streaming(main_content)
        response_placeholder.markdown(streaming_content)

    # Return the main content (without analysis or assessment), analysis, None for assessment, and thinking
    return main_content, analysis_content, None, thinking_content
