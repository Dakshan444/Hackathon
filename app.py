from flask import Flask, request, jsonify, send_file, session
import langchain
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import pandas as pd
import json
from fpdf import FPDF
import binascii

secret_key = binascii.hexlify(os.urandom(32)).decode()


load_dotenv()

app = Flask(__name__)

app.secret_key = secret_key


model = ChatGoogleGenerativeAI(
    model="gemini-pro", google_api_key=os.environ.get("GOOGLE_API_KEY"), temperature=0
)


def get_brd_text(document):
    text = ""
    page_reader = PdfReader(document)
    for page in page_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunk(text):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunk = text_spliter.split_text(text)
    return chunk


def get_vectorstore(chuncks):
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    vectorstore = FAISS.from_texts(texts=chuncks, embedding=embedding)
    return vectorstore


def get_component(text_chunks):
    prompt_template = """The following is a document
    {doc}
    Based on this doc, please identify the main product [main product] and component [list of components] with their descrption [description]
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["doc"], template=prompt_template)

    chain = LLMChain(llm=model, prompt=prompt)

    response = chain.run(doc=text_chunks)
    return response


def retreive_Component(component_with_desc):
    lines = component_with_desc.split("\n")
    components_index = lines.index("**Components:**")
    component_names = [
        line.strip("* ").split(":")[0].strip()
        for line in lines[components_index + 1 :]
        if line.startswith("*")
    ]
    return component_names


def get_user_story(component):
    prompt_template = """using following
    {text}

    Based on this text, create a user story for the buisnees for JIRA.

    When creating user stories in JIRA, it's important to include key attributes to provide clarity and context for the development team. Here are the key attributes typically used when creating user stories in JIRA:

    Summary: A brief, descriptive title for the user story that summarizes its purpose or goal.

    Description: A detailed explanation of the user story, including what needs to be done and why it's important. This should provide enough context for the development team to understand the requirements and objectives.

    Acceptance Criteria: Specific conditions or criteria that must be met for the user story to be considered complete. These criteria help define the scope of the work and provide a basis for testing.

    Priority: Indicates the relative importance or urgency of the user story compared to other items in the backlog. This helps the team prioritize their work and focus on the most valuable tasks first.

    Story Points: An estimate of the relative effort or complexity of the user story, often represented using a numerical value such as story points or ideal days. This helps the team understand the size of the task and plan their work accordingly.

    Assignee: The team member responsible for implementing the user story. Assigning tasks to specific team members helps clarify accountability and ensures that work is evenly distributed.

    Epic/Theme: If the user story is part of a larger initiative or project, it may be linked to an epic or theme in JIRA. This provides additional context and helps organize related work items.

    Labels/Tags: Optional labels or tags that provide additional categorization or metadata for the user story. This can be useful for filtering and searching for related items in the backlog.

    Linked Issues: Links to other issues or tasks that are related to the user story, such as dependencies or follow-up work. This helps ensure that the team is aware of any interconnected tasks or requirements.

    Note: Create user stories for each components from [component_names] and separate each user stories by a line like this ____.
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

    chain = LLMChain(llm=model, prompt=prompt)

    response = chain.run(text=component)
    return response


# def generate_pdf(user_stories):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for component, story in user_stories.items():
        pdf.cell(200, 10, txt=f"Component: {component}", ln=True, align="L")
        pdf.cell(200, 10, txt="", ln=True, align="L")  # Add an empty line for spacing
        for key, value in story.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align="L")
        pdf.cell(200, 10, txt="", ln=True, align="L")  # Add an empty line between user stories

    pdf_output_path = "user_stories.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path

def generate_pdf(user_stories):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for component, story in user_stories.items():
        if isinstance(story, dict):  # Check if story is a dictionary
            pdf.cell(200, 10, txt=f"Component: {component}", ln=True, align="L")
            pdf.cell(200, 10, txt="", ln=True, align="L")  # Add an empty line for spacing
            # Add user story details
            for key, value in story.items():
                pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align="L")
            pdf.cell(200, 10, txt="", ln=True, align="L")  # Add an empty line between user stories

    pdf_output_path = "user_stories.pdf"
    pdf.output(pdf_output_path)
    print(f"PDF generated successfully: {pdf_output_path}")

    return pdf_output_path


@app.route("/upload_brd", methods=["POST"])
def upload_brd():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file:
        raw_text = get_brd_text(file)
        text_chunks = get_text_chunk(raw_text)
        vector_store = get_vectorstore(text_chunks)

        components = get_component(text_chunks)
        component_name=retreive_Component(components)
        print(component_name)
        if not components:
            return jsonify({"error": "No components found in the BRD"})

        user_stories = {}
        for component in component_name:
            user_stories[component] = get_user_story(component)
        
        session["User_Stories"]=user_stories
        print(session["User_Stories"])
        print("END OF THE MODEL................")

        pdf_path = generate_pdf(user_stories)

        return send_file(pdf_path, as_attachment=True)


@app.route('/get_user_stories', methods=['GET'])
def get_user_stories():
    try:
        # Retrieve user stories from the session
        user_stories = session.get("User_Stories")
        print(session["User_Stories"])
        if user_stories:
            # Return user stories as JSON response
            return jsonify(user_stories)
        else:
            return 'No user stories found in session', 404
    except Exception as e:
        return f'Error: {str(e)}', 500



if __name__ == "__main__":
    app.run(debug=True)
