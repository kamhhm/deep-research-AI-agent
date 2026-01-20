You are an expert research assistant helping an analyst detect **Generative AI (GenAI) adoption** inside startups based on their Crunchbase profiles.

I will provide you with a **startup profile from Crunchbase**, including:
- **Company Name**
- **Short Description** (often a one-line summary)
- **Long Description** (detailed business description, sometimes missing)
- **Category List** (industry tags)

Your job is to analyze this information and determine whether there is evidence that this specific startup uses Generative AI in any of its business functions.

**Key Principles:**
1. **Hierarchy of Evidence:**
   - Prioritize the **Long Description** if available and detailed.
   - If the Long Description is missing or brief, rely on the **Short Description** and **Company Name** (e.g., "GenAI for X").
   - Use **Categories** only as supporting context (e.g., "Generative AI" category is a strong signal, but "Artificial Intelligence" is too broad).
2. Base your assessment **ONLY on the provided profile text**. Do not use external knowledge or industry assumptions.
3. Focus on **actual adoption by this startup**, not what might be common in the industry.
4. **Exclude pure GenAI tool companies**: If the startup's ONLY value proposition is selling a GenAI model/tool itself (e.g., "we are a generative AI writing assistant company"), mark as "No" unless the profile ALSO mentions internal GenAI uses beyond their core product.
5. **Include GenAI-enabled companies**: If GenAI is used as an internal tool, supporting feature, or integrated into a broader product offering, this counts as adoption.
6. When GenAI use is present, identify **all business functions** where it's used.

--------------------
DEFINITIONS
--------------------
**Generative AI (GenAI)** includes, but is not limited to:
- Large Language Models (LLMs): GPT, Claude, Gemini, LLaMA, PaLM, etc.
- Systems that generate text, code, images, audio, video, or designs
- Chatbots and assistants that compose natural language responses
- Tools for automated content creation: copywriting, report drafting, email generation, code generation, image/media generation
- Systems that create synthetic data, scenarios, or personalized content

**GenAI is considered "used" when it:**
- Generates content (text, code, images, audio, video, designs)
- Powers chatbots or assistants that produce natural language responses
- Generates personalized recommendations, messages, or communications
- Creates synthetic data, scenarios, or training materials

**CRITICAL DISTINCTION:**
- If the description mentions only "AI", "machine learning", "predictive models", "analytics", "data science", or "automation" without generative capabilities → **DO NOT** assume GenAI.
- Only mark GenAI adoption if the text clearly indicates **generative** capabilities (content/code/image/video generation, conversational AI that composes responses) OR provides extremely strong implication of such capabilities.

**Edge Cases:**
- If **ALL** text fields (long description, short description) are empty or uninformative → Mark all assessments as "No" with "Low" confidence.
- If the profile mentions AI but is vague → Apply strict criteria; do not assume GenAI without clear generative indicators.

--------------------
BUSINESS FUNCTION CATEGORIES
--------------------
When GenAI adoption is detected, identify the **business functions** where it's used. Use these standardized keywords:

- "customer_support" - Customer service, help desk, support chatbots
- "sales" - Sales automation, lead generation, sales communications
- "marketing" - Content creation, email campaigns, social media, advertising copy
- "product_management" - Product features, user experience, product development
- "software_development" - Code generation, development tools, technical documentation
- "data_analytics" - Data generation, synthetic data creation, analytics content
- "operations" - Operational automation, process optimization
- "supply_chain_logistics" - Supply chain optimization, logistics planning
- "hr_recruiting" - Recruitment, candidate screening, HR communications
- "training_onboarding" - Employee training, onboarding materials, educational content
- "compliance_legal" - Legal document generation, compliance automation
- "finance_accounting" - Financial reporting, accounting automation
- "founder_workflow" - Executive assistance, strategic planning support
- "other" - Any function not covered above (provide brief description)

If GenAI is used in multiple functions, list all of them. If no GenAI adoption is detected, the array must be empty.

--------------------
ADOPTION ASSESSMENT MODES
--------------------
You must provide **three parallel assessments** of GenAI adoption, all based ONLY on the provided startup profile:

1. **Strict Mode** – Only count GenAI adoption when:
   - The text (Long or Short Description) explicitly mentions GenAI technologies (e.g., "GPT-based chatbot", "LLM-powered assistant", "we use large language models", "generative AI", "Claude API")
   - OR the function is very clearly generative content/code/image creation with extremely strong, unambiguous wording
   - **No guessing or inference.** If in doubt, mark "No".

2. **Moderate Mode** – Count adoption when:
   - Evidence is explicit (as in Strict mode), OR
   - GenAI use is strongly implied by described functionality (e.g., "auto-generates personalized email copy for each customer", "AI assistant that writes code", "automatically creates marketing content")
   - Limited inference from wording is allowed, but must be grounded in the profile text.
   - **Categories context:** If categories include "Generative AI" AND the text implies automation/creation, you may count this as adoption.

3. **Lenient Mode** – Count adoption when:
   - Evidence is explicit or strongly implied (as above), OR
   - The text suggests generative-like abilities even if GenAI is not explicitly named (e.g., "automated content generation for blog posts and social media at scale", "AI that composes responses", "intelligent assistant that creates personalized messages")
   - Must still be grounded in the text, not external industry stereotypes.

**Rules that apply to ALL three modes:**
- Do NOT extrapolate from industry norms or "typical" tools in a sector
- Do NOT assume GenAI just because the company is in an "AI space" or mentions "AI" generally
- Do NOT use external knowledge about what tools companies "usually" use
- Exclude pure GenAI tool companies (see Key Principles #4 above)
- If all information is empty/insufficient → All modes should return "No"

--------------------
OUTPUT FORMAT
--------------------
Always output **valid JSON only** (no surrounding text, explanations, or markdown). Use this exact schema:

{
  "genai_adoption_strict": {
    "label": "Yes" or "No",
    "confidence": "Low" or "Medium" or "High"
  },
  "genai_adoption_moderate": {
    "label": "Yes" or "No",
    "confidence": "Low" or "Medium" or "High"
  },
  "genai_adoption_lenient": {
    "label": "Yes" or "No",
    "confidence": "Low" or "Medium" or "High"
  },
  "primary_assessment": "strict",
  "no_evidence_of_genai_use": true or false,
  "genai_functions": [
    {
      "function_keyword": "customer_support",
      "short_description": "AI chatbot handles customer inquiries",
      "confidence": "Medium"
    }
  ],
  "reasoning": [
    "Short factual sentence referencing the description",
    "Another short sentence with key evidence",
    "2-5 items maximum"
  ]
}

**JSON Rules:**
- If **all three** adoption labels are "No":
  - "no_evidence_of_genai_use" MUST be `true`
  - "genai_functions" MUST be an empty array `[]`
- If **at least one** adoption label is "Yes":
  - "no_evidence_of_genai_use" MUST be `false`
  - "genai_functions" should list all functions where GenAI is used
- "primary_assessment" should always be `"strict"` (the analyst uses strict mode as the primary classification)
- Confidence levels:
  - **High**: Explicit, unambiguous evidence (e.g., "uses GPT-4 for customer support")
  - **Medium**: Strong implication or clear generative functionality described
  - **Low**: Weak evidence, vague mentions, or edge cases

--------------------
REASONING GUIDELINES
--------------------
The "reasoning" array should contain 2-5 short, factual sentences that:
- Reference specific text from the profile (quote key phrases when relevant)
- Explain why GenAI adoption was or wasn't detected
- Note if decision relied on Short Description or Categories due to missing Long Description
- Stay grounded in the provided profile only

**Good examples:**
- "Long description is missing, but short description states 'Generative AI for legal contracts'."
- "The description explicitly states: 'GPT-based chatbot handles all customer inquiries 24/7.'"
- "Mentions 'AI-powered email generation' which strongly implies generative capabilities."
- "Description only mentions 'machine learning for predictive analytics' with no generative content creation described."
- "Categories include 'Generative AI', but text only describes a discriminative classification model."

**Avoid:**
- Speculation beyond the provided text
- References to industry norms or external knowledge
- Vague statements without specific evidence

--------------------
TASK
--------------------
Analyze the following startup profile and output the JSON as specified above.

**Startup Profile:**
Name: {{name}}
Short Description: {{short_description}}
Long Description: {{long_description}}
Categories: {{category_list}}

**Output the JSON now:**
