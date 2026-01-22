# AI-Rubric Writer - User Guide

A collaborative writing tool that helps you develop personalized rubrics and improve your writing with AI assistance.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Projects](#projects)
3. [Chat Tab](#chat-tab)
4. [Branch Mode](#branch-mode)
5. [View Rubric Tab](#view-rubric-tab)
6. [Compare Rubrics Tab](#compare-rubrics-tab)
7. [Evaluate: Coverage Tab](#evaluate-coverage-tab)
8. [Evaluate: Alignment Tab](#evaluate-alignment-tab)
9. [Evaluate: Utility Tab](#evaluate-utility-tab)

---

## Getting Started

### First Time Setup
1. Open the app in your browser
2. A default project will be loaded automatically
3. You can start chatting right away or create a new project

### Interface Overview
- **Sidebar (left)**: Project selection, rubric configuration, and settings
- **Main area (right)**: Six tabs for different features

---

## Projects

Projects keep your work organized. Each project contains:
- Conversation history
- Rubric versions
- Survey responses

### Create a New Project
1. In the sidebar, click **"➕ Create New Project"**
2. Enter a project name (use letters, numbers, hyphens, or underscores)
3. Click **"Create Project"**

### Switch Between Projects
1. Use the **"Select Project"** dropdown in the sidebar
2. Your conversations and rubrics will load automatically

### Export a Project
1. In the sidebar, click **"📦 Export / Import Project"**
2. Click **"📥 Export '{project-name}'"**
3. A ZIP file will download containing all your project data

### Import a Project
1. In the sidebar, click **"📦 Export / Import Project"**
2. Click **"Browse files"** and select a previously exported ZIP file
3. Click **"📤 Import Project"**
4. The project will be imported and selected automatically

---

## Chat Tab

The Chat tab is where you collaborate with AI to improve your writing.

### Starting a Conversation
1. Select **"New Conversation"** from the dropdown (or select an existing one)
2. Type your writing prompt or paste your draft in the text box
3. Press Enter or click the send button

### Working with the AI
The AI can help you:
- **Write drafts** based on your prompts
- **Revise existing writing** based on feedback
- **Assess your draft** against the rubric
- **Infer rubric criteria** from your conversation

### Editable Drafts
When the AI generates a draft (wrapped in `<draft></draft>` tags), it appears as an editable text area. You can:

1. **Edit the draft** directly in the text area
2. **💾 Save**: Save your edits as a new message in the conversation
3. **🔄 Update Rubric**: Ask the AI to analyze your changes and suggest rubric updates
4. **↩️ Reset**: Revert to the original draft

This feature lets you fine-tune the AI's output and helps the system learn your preferences.

### Action Buttons
Below the chat input, you'll find action buttons:
- **💾 Save Conversation**: Save your chat history (appears when you have messages)
- **🔍 Infer Rubric**: Generate rubric criteria from your conversation preferences
- **📊 Assess Draft**: Get rubric-based feedback on your most recent draft (requires a draft wrapped in `<draft></draft>` tags)
- **🌿 Create Branch**: Start a temporary branch to experiment with changes (see [Branch Mode](#branch-mode))
- **🗑️ Clear**: Start a new conversation (clears current messages)

### Rubric Assessment Display
After clicking **"📊 Assess Draft"**, you'll see:
- An expandable **"📊 Rubric Assessment"** card
- Each criterion shows its score and detailed feedback
- **Evidence highlighting** shows which parts of your draft relate to each criterion
- You can provide feedback on the assessment for each criterion

### AI Thinking
Many AI responses include a **"🧠 Thinking"** expander that shows the AI's reasoning process. Click to expand and see how the AI arrived at its response.

### Saving Conversations
- Click **"💾 Save Conversation"** to save your chat history
- Saved conversations appear in the dropdown for later access

---

## Branch Mode

Branch mode allows you to experiment with rubric changes and continue conversations without affecting your main work. Think of it as a "sandbox" for trying out ideas.

### When to Use Branch Mode
- Testing how different rubric weights affect AI responses
- Experimenting with new rubric criteria before committing
- Exploring alternative directions in a conversation
- Trying out changes without risk

### Creating a Branch
1. In the Chat tab, click **"🌿 Create Branch"**
2. The sidebar will show **"🌿 Branch Mode Active"** indicator
3. The chat input will show "🌿 Branch message (temporary)..."

### Working in Branch Mode

#### Branch Messages
- Messages you send in branch mode appear in a separate **"🌿 Branch Messages"** section
- These messages are marked as temporary
- The AI has access to both main conversation and branch messages

#### Branch Rubric
When in branch mode, the sidebar shows a **"📝 Branch Rubric"** section where you can:

1. **Edit Weights**: Adjust the importance of each criterion using sliders
2. **Edit Content**: Modify descriptions, achievement levels, and dimensions
3. **Ask for Suggestions**: Use the chat box to ask the AI for rubric improvements
   - Example: "Make clarity more important" or "Add a criterion for creativity"
4. **Apply Suggestions**: Click **"✨ Update Rubric with Suggestions"** to apply AI recommendations
5. **Regenerate Draft**: Click **"✨ Regenerate Draft"** to see how your rubric changes affect the AI's writing

### Exiting Branch Mode

You have two options:

#### Merge Branch
Click **"✅ Merge Branch to Main"** to:
- Add all branch messages to your main conversation
- Save your modified rubric as a new version
- Return to normal mode

#### Discard Branch
Click **"❌ Discard Branch"** to:
- Delete all branch messages
- Discard any rubric changes
- Return to your original state

### Tips for Branch Mode
- Use branches to A/B test different rubric configurations
- Make multiple changes at once, then regenerate the draft to see the combined effect
- If you like the results, merge; if not, discard and try again
- The **"🔄 Reset"** button in the sidebar resets only the rubric changes (keeps messages)

---

## View Rubric Tab

View and manage your rubric criteria.

### Weight Distribution Chart
At the top of the tab, a **pie chart** shows how weights are distributed across your rubric criteria. This helps you visualize which aspects of writing you're prioritizing.

### Understanding the Rubric
Each criterion has:
- **Name**: What aspect of writing it evaluates
- **Weight**: How important it is (as a percentage)
- **Category**: Grouping for related criteria
- **Achievement Levels**: Four levels from Beginning to Exemplary

### Rubric Versions
- Each time you modify the rubric, a new version is created
- Use the **"Active Version"** dropdown in the sidebar to switch between versions
- View version history with timestamps

### Editing the Rubric
1. In the sidebar under **"📋 Rubric Configuration"**, expand a criterion
2. Modify the name, weight, or achievement level descriptions
3. Click **"💾 Save Changes"** to create a new version

### Adding/Removing Criteria
- Click **"➕ Add Criterion"** to add a new criterion
- Click the **"🗑️"** button next to a criterion to remove it

---

## Compare Rubrics Tab

Compare different versions of your rubric side-by-side and see how they affect writing.

### How to Compare
1. Select two rubric versions using the dropdowns
2. View the differences highlighted:
   - **‼️ Icon**: New or changed criterion
   - Expand each criterion to see detailed changes

### Generate a Writing Comparison
To see how different rubrics affect actual writing:

1. Enter a **writing task** in the text area (e.g., "Write a professional email declining a meeting")
2. Click **"Generate Comparison"**
3. View the results:
   - **Key Differences**: Summary of how the rubrics differ
   - **Summary**: Impact analysis of the differences
   - **Base Draft**: The same starting draft
   - **Rubric A Revision**: Draft revised according to Rubric A
   - **Rubric B Revision**: Draft revised according to Rubric B

Changes between versions are highlighted in green (additions) and red (deletions).

### Use Cases
- See how your rubric evolved over time
- Understand what changed between versions
- **See the practical impact** of rubric changes on actual writing
- Decide which version better captures your preferences

---

## Evaluate: Coverage Tab

Test whether your rubric accurately predicts your writing preferences.

### Step 1: Select a Conversation
1. Choose a conversation from the dropdown
2. Click **"🎯 Extract Decision Points & Generate Reflections"**
3. The AI analyzes places where you edited or changed the AI's suggestions

### Step 2: Review Decision Points
The conversation view highlights relevant messages. For each decision point:

1. Click on a decision point button to expand it
2. See the **Before** (AI's suggestion) and **After** (your edit)
3. View the AI's analysis of why you might have made this change

### Step 3: Reflect on Your Preferences
For each decision point, answer three questions:
1. **What made you want to change this?** - Describe your motivation
2. **What was wrong with the original?** - Explain the specific issues
3. **Is this a general preference?** - Choose: General / Specific / Depends on context

Click **"💾 Save Reflection"** for each one.

### Step 4: Generate Preference Tests
1. Click **"✨ Generate Preference Tests"**
2. The AI extracts preference dimensions from your reflections
3. For each test:
   - Compare **Version 1** and **Version 2**
   - Select which you prefer
   - Explain your reasoning

### Step 5: Validate the Rubric
1. Click **"📏 Test with Rubric"**
2. The rubric scores each test's versions
3. View the results:
   - **Prediction Accuracy**: How often the rubric correctly predicted your preference
   - **Coverage**: How many preference dimensions the rubric addresses
   - Detailed breakdown by test showing matches and mismatches

### Saving Results
After completing the analysis, you have two options:

- **💾 Save to Project**: Saves to `evaluate_coverage.json` in your project folder. Multiple analyses are appended to the same file.
- **📥 Download as JSON**: Downloads a timestamped file to your browser's download folder.

**Data saved includes:**
- Conversation analyzed
- Rubric version used
- Decision points extracted
- Your reflections for each decision point
- Preference dimensions identified
- Test comparisons generated
- Your preference test responses
- Rubric validation scores and predictions

### Reset
Click **"🔄 Reset Coverage Test"** to start over with a fresh evaluation.

---

## Evaluate: Alignment Tab

Test whether you and the AI interpret the rubric criteria the same way.

### Step 1: Select a Draft
1. Choose a conversation from the dropdown
2. Select a specific draft to evaluate (drafts must be wrapped in `<draft></draft>` tags)
3. Click **"View full draft"** to review the complete text

### Step 2: Score the Draft
1. A progress bar shows how many criteria you've scored
2. For each rubric criterion:
   - Review the achievement level descriptions (Exemplary, Proficient, Developing, Beginning)
   - Select the level that best matches the draft
3. Once all criteria are scored, the comparison button becomes available

### Step 3: Compare with AI
1. Click **"🎯 Get AI Scores & Compare"**
2. The AI scores the same draft using the same rubric
3. View the alignment results:

### Alignment Metrics
- **Exact Agreement**: Number of criteria where you gave the exact same score
- **Rank Correlation (ρ)**: Spearman correlation showing how well rankings match (-1 to 1)
- **Avg Score Distance**: Average difference in levels between your scores and AI's
- **Systematic Bias**: Whether AI tends to score higher or lower than you

### Interpretation Guide
- **≥70% exact agreement**: Strong alignment - you and AI interpret similarly
- **50-70% exact agreement**: Moderate alignment - consider clarifying rubric language
- **<50% exact agreement**: Low alignment - rubric may be ambiguous

### Detailed Comparison
For each criterion:
- **✅** indicates agreement, **❌** indicates disagreement
- View AI's rationale for its score
- See specific evidence the AI cited from the draft

### Evidence Highlights
At the bottom, view the draft with **color-coded highlighting**:
- Different colors for different criteria
- **Hover over highlighted text** to see:
  - Which criterion it relates to
  - Why the AI found it relevant

This helps you understand exactly how the AI interprets your rubric criteria.

### Saving Results
After the comparison is complete, you have two options:

- **💾 Save to Project**: Saves to `evaluate_alignment.json` in your project folder. Multiple analyses are appended to the same file.
- **📥 Download as JSON**: Downloads a timestamped file to your browser's download folder.

**Data saved includes:**
- Conversation and draft analyzed
- Draft content
- Rubric version used
- Alignment metrics (exact agreement, correlation, bias, etc.)
- For each criterion:
  - Your score and the AI's score
  - Whether they matched
  - AI's rationale and evidence quotes
- Evidence highlights with positions

### Reset
Click **"🔄 Reset Alignment Test"** to start over with a fresh evaluation.

---

## Evaluate: Utility Tab

Measure the usefulness of your rubric through surveys.

### Survey Structure
Two conditions to compare:
1. **Without Rubric**: Complete surveys after writing WITHOUT the rubric
2. **With Rubric**: Complete surveys after writing WITH the rubric

### Cognitive Load Survey
Measures mental effort during writing:
- Mental Demand
- Physical Demand
- Temporal Demand
- Effort
- Performance
- Frustration

### Writing Quality Survey
Measures your satisfaction with the writing:
- Content Satisfaction
- Ownership
- Responsibility
- Personal Connection
- Emotional Connection
- Prompt Difficulty

### How to Use
1. Write something **without** using the rubric
2. Complete both surveys in the **"WITHOUT RUBRIC"** tab
3. Write something similar **with** the rubric
4. Complete both surveys in the **"WITH RUBRIC"** tab
5. Compare your responses to see if the rubric helps

### Saving Survey Data
Survey responses are automatically saved to your project folder when you click **"Submit"**:
- `cognitive_load_responses.json` - Cognitive load survey data
- `writing_quality_responses.json` - Writing quality survey data

### Reset
Click **"🔄 Reset Utility Test"** to clear all survey responses and start fresh.

---

## Tips for Best Results

### Building a Good Rubric
1. Start by writing naturally and chatting with the AI
2. Use "infer rubric" to generate initial criteria
3. Refine the rubric based on your preferences
4. Test it with the Evaluate tabs

### Iterating on Your Rubric
1. Use **Evaluate: Coverage** to check if the rubric captures your preferences
2. Use **Evaluate: Alignment** to ensure consistent interpretation
3. Modify criteria that have low agreement or coverage
4. Save new versions and compare them

### Saving Your Work
- **Save conversations** regularly to preserve your chat history
- **Export your project** before closing to backup everything
- **Import projects** on a new device to continue working

---

## Troubleshooting

### "No project selected" error
- Select a project from the sidebar dropdown
- Or create a new project

### API errors
- Check that your API key is configured correctly
- Ensure you have sufficient API credits

### Lost data after refresh (on Streamlit Cloud)
- Use the **Export Project** feature to save your work locally
- Import it when you return to continue


