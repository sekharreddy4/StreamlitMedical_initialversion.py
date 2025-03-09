import os
from typing import List
from PyPDF2 import PdfReader
import streamlit as st
from langchain_groq import ChatGroq  # Correct import for ChatGroq
from langchain.prompts import PromptTemplate  # Correct import for PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory  # Updated memory system
from langchain.schema.messages import AIMessage, HumanMessage  # For handling messages

# Set your Groq API key (ensure it is set as an environment variable)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API key
groqapikey = os.getenv("api-key")
if not groqapikey:
    st.error("Error: API key not found in .env file.")
    st.stop()


# Initialize the LLM
llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.3-70b-versatile",
    api_key=groqapikey
)

def parse_medical_documents(file_path: str) -> str:
    """Parse medical documents from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        if not text.strip():
            raise ValueError("No text could be extracted from the document.")
        return text
    except FileNotFoundError:
        return "No medical document available."
    except Exception as e:
        return f"Error while parsing the document: {e}"

class ReasoningEngine:
    def __init__(self, llm, medical_data: str):
        self.llm = llm
        self.medical_data = medical_data

    def diagnose(self, symptoms: List[str]) -> str:
        """Diagnose based on symptoms and medical knowledge."""
        prompt = f"""
        Based on the following symptoms: {', '.join(symptoms)}, and the medical knowledge extracted, what are the possible diagnoses?
        Provide a concise explanation for each possible diagnosis.

        Medical Knowledge: {self.medical_data[:2000]}  # Truncated to first 2000 characters
        """
        try:
            response = self.llm.invoke(prompt)  # Updated to use `invoke` method for LLMs
            if isinstance(response, AIMessage):
                return response.content  # Extract content from AIMessage object
            return str(response)
        except Exception as e:
            return f"Error during diagnosis: {e}"

# Define Prompt Templates
initial_prompt_template = """
You are a helpful medical assistant. Your goal is to gather information from the patient to help them determine the possible cause of their symptoms. Start by asking the patient what their primary complaint is.

Current conversation:
{history}
Patient: {input}
AI: """

followup_prompt_template = """
You are a helpful medical assistant. You have collected the following information:
{context}

Based on the information, what is the next most relevant question to ask the patient to narrow down potential diagnoses? Be specific and ask only one question. Explain why you are asking this question.

Current conversation:
{history}
Patient: {input}
AI: """

initial_prompt = PromptTemplate(input_variables=["history", "input"], template=initial_prompt_template)
followup_prompt = PromptTemplate(input_variables=["history", "input", "context"], template=followup_prompt_template)

# Initialize Memory (New System)
message_history = ChatMessageHistory()

def save_message_to_history(user_input, ai_response):
    """Save messages to chat history."""
    message_history.add_user_message(user_input)
    message_history.add_ai_message(ai_response)

def get_conversation_history():
    """Retrieve conversation history as a formatted string."""
    messages = message_history.messages
    history = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            history += f"Patient: {message.content}\n"
        elif isinstance(message, AIMessage):
            history += f"AI: {message.content}\n"
    return history

# Streamlit App Interface
st.title("AI Medical Assistant")
st.write("Describe your symptoms, and I will help you identify potential causes.")

# Initialize session state for context and symptoms
if "context" not in st.session_state:
    st.session_state.context = ""
if "symptoms" not in st.session_state:
    st.session_state.symptoms = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store full conversation history

# Create a form for user input
with st.form("symptom_form", clear_on_submit=True):
    user_input = st.text_input("Enter your symptom or complaint:")
    submitted = st.form_submit_button("Submit")

if submitted:
    if user_input.strip():
        # Get conversation history for context
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])

        # Determine whether it's the first interaction or follow-up
        if not context:  # First interaction
            ai_response_object = (initial_prompt | llm).invoke({"history": context, "input": user_input})
        else:
            ai_response_object = (followup_prompt | llm).invoke({
                "history": context,
                "input": user_input,
                "context": context,
            })

        # Extract AI response content (handle both AIMessage and other formats)
        ai_response_content = (
            ai_response_object.content if isinstance(ai_response_object, AIMessage) else str(ai_response_object)
        )

        # Save messages to chat history in session state
        st.session_state.chat_history.append({"role": "Patient", "content": user_input})
        st.session_state.chat_history.append({"role": "AI", "content": ai_response_content})

        # Display AI response
        st.write(f"**AI:** {ai_response_content}")

        # Save symptoms in session state
        st.session_state.symptoms.append(user_input)

        # Diagnose after collecting at least 3 symptoms
        if len(st.session_state.symptoms) >= 3:
            with st.spinner("Analyzing symptoms..."):
                engine = ReasoningEngine(llm=llm, medical_data="Sample medical knowledge")
                diagnosis = engine.diagnose(st.session_state.symptoms)
                st.success("Diagnosis Complete!")
                st.write(f"### Potential Diagnosis:\n{diagnosis}")
                # Clear symptoms after diagnosis if you want to reset.
                st.session_state.symptoms.clear()
    else:
        st.warning("Please enter a symptom or complaint.")

# Display full conversation history (optional)
st.write("### Conversation History:")
for message in st.session_state.chat_history:
    role_prefix = "Patient" if message["role"] == "Patient" else "AI"
    st.write(f"**{role_prefix}:** {message['content']}")
