pip install --upgrade pip
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores.cassandra import Cassandra
# Replace the old import with the new one
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import cassio
from dotenv import load_dotenv
from gtts import gTTS  # Import Google Text-to-Speech
from sentence_transformers import SentenceTransformer
from langchain.prompts.chat import ChatPromptTemplate
import io  # Import io for in-memory file operations

# Load environment variables
load_dotenv()

# Access environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

# Streamlit page configuration
st.set_page_config(page_title="Annie - The Healthcare Advocate", layout="wide")
st.title("🏥 Annie - Your AI Guide to Non-Profit Hospitals")

# Main Introduction
st.image("annie_avatar.png", width=150)  # Display AI Avatar (ensure you have this image)
st.subheader("Hello, I’m Annie! 🌟")
intro_text = "I'm here to help you understand why hospitals should be non-profit. Ask me anything, and I'll provide answers in both text and voice formats!"
st.write(intro_text)

# Play introductory text as audio
intro_audio = gTTS(intro_text, lang='en')
intro_buffer = io.BytesIO()
intro_audio.write_to_fp(intro_buffer)
intro_buffer.seek(0)
st.audio(intro_buffer, format="audio/mp3")

# Memory to store conversation
memory = []  # List to store previous questions and answers

# Sidebar: Writer Info, Creator Info, FAQ, Read Article, and Feedback
with st.sidebar:
    # Stylize Features section header
    st.markdown("<h2 style='color: #FF5733; font-weight: bold;'>🔧 Features</h2>", unsafe_allow_html=True)

    # Writer Info Button
    if st.button("👩‍⚕️ Writer Info"):
        writer_info_text = (
            "The author of this insightful piece is Qurat-ul-Ain Bhatti, a dedicated medical student at the "
            "University of Management and Technology (UMT). With her background in healthcare and her passion for "
            "social impact, Qurat-ul-Ain has thoughtfully explored why hospitals should prioritize patient care over profit.\n\n"
            "Her idea of making hospitals non-profit reflects not only her deep understanding of the medical field but also "
            "her genuine compassion for patients and communities. It’s rare to see such kindness and commitment from someone "
            "still in their medical training, and it’s inspiring to see her advocate for a healthcare model that serves everyone equally. "
            "We commend her for taking a stand on such an important issue and for using her voice to make a positive difference in the world. 🌟"
        )
        st.session_state.writer_info_displayed = True  # Store display state
        st.session_state.writer_info_text = writer_info_text

    # Creator Info Button
    if st.button("👨‍💻 Creator Info"):
        creator_info_text = (
            "This chatbot was created by Muhammad Adnan, a passionate Electrical Engineering student in his 7th semester "
            "at the National University of Sciences and Technology (NUST). Driven by a genuine admiration for Qurat-ul-Ain’s compassionate vision, "
            "Adnan built this platform to amplify her message and advocate for accessible, patient-centered healthcare.\n\n"
            "With a strong foundation in technology and a mind inspired by meaningful ideas, Adnan designed this chatbot to bridge the gap between complex "
            "healthcare concepts and everyday understanding. His work not only showcases his technical skills but also his respect for humanitarian values. "
            "By blending engineering with empathy, Adnan is using his talents to bring insightful discussions to life and make a positive impact on important social issues."
        )
        st.session_state.creator_info_displayed = True  # Store display state
        st.session_state.creator_info_text = creator_info_text

    # FAQ Button
    if st.button("📖 FAQ"):
        st.session_state.show_faq = True
    
    # Read Article Button
    if st.button("📚 Read Article"):
        st.markdown("[Read the Article on Non-Profit Hospitals](https://medium.com/@s2023241029/all-hospitals-must-be-non-profit-institutions-4e79406721de)", unsafe_allow_html=True)

    # Feedback Section
    st.markdown("<h3 style='color: #007BFF;'>Feedback:</h3>", unsafe_allow_html=True)
    feedback = st.radio("How was your experience?", ["👍 Good Work", "👎 Needs Improvement"])
    
    if feedback == "👍 Good Work":
        st.success("Thank you for your feedback! We're glad you're satisfied! 😊")
    elif feedback == "👎 Needs Improvement":
        st.warning("Thank you for your feedback! We’ll work to improve! 🙏")
    
    # Additional Resource Links for a fuller sidebar
    st.markdown("<h3 style='color: #007BFF;'>More Resources:</h3>", unsafe_allow_html=True)
    st.markdown("- [WHO on Health Systems](https://www.who.int/health-topics/health-systems)")
    st.markdown("- [Harvard Health Blog](https://www.health.harvard.edu/blog)")

# Display Writer Info or Creator Info if button was clicked
if "writer_info_displayed" in st.session_state and st.session_state.writer_info_displayed:
    st.header("About the Writer")
    st.write(st.session_state.writer_info_text)
    writer_audio = gTTS(st.session_state.writer_info_text, lang='en')
    writer_audio_buffer = io.BytesIO()
    writer_audio.write_to_fp(writer_audio_buffer)
    writer_audio_buffer.seek(0)
    st.audio(writer_audio_buffer, format="audio/mp3")
    st.session_state.writer_info_displayed = False  # Reset display state

if "creator_info_displayed" in st.session_state and st.session_state.creator_info_displayed:
    st.header("About the Creator")
    st.write(st.session_state.creator_info_text)
    creator_audio = gTTS(st.session_state.creator_info_text, lang='en')
    creator_audio_buffer = io.BytesIO()
    creator_audio.write_to_fp(creator_audio_buffer)
    creator_audio_buffer.seek(0)
    st.audio(creator_audio_buffer, format="audio/mp3")
    st.session_state.creator_info_displayed = False  # Reset display state

# Load the Hugging Face model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the embedding wrapper for Cassandra VectorStore
class HuggingFaceEmbeddingWrapper:
    def __init__(self, model):
        self.model = model
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()
    
    def embed_documents(self, texts):
        return [self.model.encode([text])[0].tolist() for text in texts]

# Initialize the embedding wrapper
embedding_wrapper = HuggingFaceEmbeddingWrapper(model)

# Initialize Cassandra session using cassio
session = cassio.init(
    token=ASTRA_DB_APPLICATION_TOKEN,
    database_id=ASTRA_DB_ID
)

# Initialize Cassandra VectorStore for querying
astra_vector_store = Cassandra(
    embedding=embedding_wrapper,
    table_name="qa_mini_demo",
    session=session,
    keyspace=None  # Replace with your actual keyspace if needed
)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Initialize Gemini (Google Generative AI) LLM for Chat
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=512,
    max_retries=2,
    timeout=None
)

# Create a prompt template for the LLM to generate answers based on the context and memory
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that remembers previous conversations. Use the following memory: {memory}. "
            "Here is the context: {context}. Answer the user's question.",
        ),
        ("human", "{question}"),
    ]
)

# Combine LLM and prompt template for querying (Define the chain)
chain = prompt_template | llm

# Function to clean the response and remove metadata
def clean_response(response):
    return response.content.replace('\n', ' ').strip()

# Function to retrieve relevant documents from the database
def get_relevant_text_from_db(query):
    relevant_docs = astra_vector_store.similarity_search(query, k=3)
    if len(relevant_docs) == 0:
        st.warning("No relevant documents were retrieved from Astra DB.")
        return ""
    combined_context = " ".join([doc.page_content for doc in relevant_docs])
    return combined_context

# Function to generate speech using gTTS and play it directly
def generate_audio(text):
    tts = gTTS(text, lang='en')
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    st.audio(audio_buffer, format="audio/mp3")

# FAQ Questions
faq_questions = {
    "Why should hospitals be non-profit?": "Non-profit hospitals prioritize patient care over profit, ensuring that everyone has access to essential healthcare without financial burdens.",
    "Does a non-profit hospital model improve patient care?": "Studies indicate that non-profit hospitals can improve patient care by focusing resources on healthcare rather than profit generation.",
    "How are non-profit hospitals funded if they don’t make profits?": "Non-profit hospitals often receive funding from donations, government grants, and other public and private sources to sustain their services.",
    "What are the benefits of non-profit hospitals for the community?": "Non-profit hospitals provide affordable healthcare, prioritize patient needs, and contribute to healthier, more resilient communities."
}

# Function to display FAQ content in the main window
def display_faq():
    st.header("Frequently Asked Questions")
    for question, answer in faq_questions.items():
        with st.expander(question):
            st.write(answer)
            generate_audio(answer)

if "show_faq" in st.session_state and st.session_state.show_faq:
    display_faq()

# Q&A Section
st.header("Ask a Question")
user_question = st.text_input("Enter your question about non-profit hospitals:")

def ask_question(question):
    context_from_db = get_relevant_text_from_db(question)
    if not context_from_db:
        return "No context retrieved from the database."
    
    memory_context = " ".join(memory)
    response = chain.invoke({
        "memory": memory_context,
        "context": context_from_db,
        "question": question
    })
    cleaned_answer = clean_response(response)
    memory.append(f"Q: {question}")
    memory.append(f"A: {cleaned_answer}")
    return cleaned_answer

if st.button("🧠 Ask Annie"):
    if user_question:
        answer = ask_question(user_question)
        st.write(f"**Annie's Answer:** {answer}")
        generate_audio(answer)
    else:
        st.warning("Please enter a question before asking Annie.")

st.markdown("---")
st.markdown("This AI-powered assistant provides insights on why hospitals should be non-profit, advocating for patient-centered healthcare.")
