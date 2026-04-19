"""Tab UI — extracted from git HEAD app.py (monolith)."""
from rubric_writer.ui._deps import *


def render_view_rubric_tab():
    st.subheader("📁 View Rubric")
    st.markdown("View the active rubric version with all criteria and checkable dimensions.")

    # Get active rubric
    active_rubric_dict, active_idx, rubric_history = get_active_rubric()

    if not active_rubric_dict or not active_rubric_dict.get("rubric"):
        st.warning("No active rubric available. Please create or select a rubric first.")
    else:
        # Display version, writing type, and user goals at the top
        col_meta1, col_meta2, col_meta3 = st.columns([1, 1, 2])

        with col_meta1:
            version = active_rubric_dict.get("version", 1)
            st.metric("Version", f"v{version}")

        with col_meta2:
            pass  # Source label removed — version number is sufficient

        with col_meta3:
            writing_type = active_rubric_dict.get("writing_type", "Not specified")
            st.markdown(f"**Writing Type:** {writing_type}")

        user_goals = active_rubric_dict.get("user_goals_summary", "")
        if user_goals:
            st.markdown("**User Goals:**")
            st.info(user_goals)

        # Achievement level explanation
        st.markdown("**Achievement Levels** *(based on dimensions met)*")
        st.markdown("⭐⭐⭐ **Excellent**: 90%+ | ⭐⭐ **Good**: 75-89% | ⭐ **Fair**: 50-74% | ◇ **Needs Work**: 25-49% | ☆ **Weak**: <25%")

        st.markdown("---")

        # Prepare rubric list
        rubric_list = active_rubric_dict.get("rubric", [])

        # Display priority rankings as a simple numbered list
        if rubric_list:
            # Sort by priority (1 = most important = shown first)
            sorted_criteria = sorted(rubric_list, key=lambda c: c.get('priority', c.get('weight', 99)))

            st.markdown("### Priority Rankings")
            for criterion in sorted_criteria:
                priority = criterion.get('priority', criterion.get('weight', 0))
                name = criterion.get('name', 'Unnamed')
                st.markdown(f"**{priority}.** {name}")

            st.markdown("---")

        # Group criteria by category
        from collections import defaultdict
        categories = defaultdict(list)

        for criterion in rubric_list:
            category = criterion.get('category', 'Uncategorized')
            categories[category].append(criterion)

        # Display each category group
        for category_name, criteria in categories.items():
            # Category header box
            st.markdown(f"""
                <div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px; margin-bottom: 16px; background-color: rgba(33, 150, 243, 0.1);">
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #2196F3; text-transform: capitalize; text-align: center;">{category_name}</div>
            """, unsafe_allow_html=True)

            # Display criteria within this category
            for criterion in criteria:
                criterion_name = criterion.get('name', 'Unnamed Criterion')
                priority = criterion.get('priority', criterion.get('weight', 0))
                description = criterion.get('description', 'No description provided')

                # Display criterion name, priority, and description
                st.markdown(f"**{criterion_name}**")
                st.markdown(f"*Priority: #{priority}*")
                st.markdown(f"{description}")

                # Expander for dimensions (checkable items)
                dimensions = criterion.get('dimensions', [])
                if dimensions:
                    with st.expander(f"Dimensions", expanded=False):
                        for dim in dimensions:
                            dim_label = dim.get('label', 'Unnamed dimension')
                            st.markdown(f"• {dim_label}")
                else:
                    # Fallback for old rubrics with achievement levels
                    exemplary = criterion.get('exemplary', '')
                    if exemplary:
                        with st.expander("📊 Achievement Levels", expanded=False):
                            proficient = criterion.get('proficient', 'Not specified')
                            developing = criterion.get('developing', 'Not specified')
                            beginning = criterion.get('beginning', 'Not specified')

                            st.markdown("**🌟 Exemplary**")
                            st.markdown(exemplary)
                            st.markdown("")

                            st.markdown("**✅ Proficient**")
                            st.markdown(proficient)
                            st.markdown("")

                            st.markdown("**📈 Developing**")
                            st.markdown(developing)
                            st.markdown("")

                            st.markdown("**🔰 Beginning**")
                            st.markdown(beginning)

                st.markdown("<br>", unsafe_allow_html=True)

            # Close the category box
            st.markdown("</div>", unsafe_allow_html=True)
