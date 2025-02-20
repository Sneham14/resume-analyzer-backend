from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

# ✅ Load AI Model
llm = ChatOpenAI(model="gpt-4o")

def expand_abbreviations_with_ai(skill):
    """
    Uses GPT-4o to dynamically expand abbreviations in skills.
    """
    prompt = PromptTemplate(
        input_variables=["skill"],
        template="""
        Expand the following abbreviation or acronym if it represents a professional skill:
        {skill}
        
        If it's a valid abbreviation, return the **full form only** as plain text.
        If it's not an abbreviation, return the same skill unchanged.
        """
    )

    chain = prompt | llm

    try:
        response = chain.invoke({"skill": skill})

        expanded_skill = response.content.strip()

        # ✅ If expansion is different from input, return the expanded version
        if expanded_skill.lower() != skill.lower():
            print(f"🔄 Expanded '{skill}' → '{expanded_skill}'")
            return expanded_skill

        return skill  # No change if not an abbreviation

    except Exception as e:
        print(f"❌ Error expanding abbreviation: {e}")
        return skill  # Return original skill if LLM fails
