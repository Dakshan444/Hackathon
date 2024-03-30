import langchain
import streamlit as st
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



load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.environ.get("GOOGLE_API_KEY")
model=ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.environ.get("GOOGLE_API_KEY"),temperature=0)

def get_brd_text(document):
    text=""
    page_reader= PdfReader(document)
    for page in page_reader.pages:
        text+=page.extract_text()
    return text

def get_text_chunk(text):
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunk=text_spliter.split_text(text)
    return chunk

def get_chunk(document):
    pdf_loader=PyPDFLoader(document)
    pdf=pdf_loader.load()
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunk=text_spliter.split_documents(pdf)
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GOOGLE_API_KEY"))
    vectorstore=FAISS.from_texts(text=chunk,embedding=embedding)
    return vectorstore

def get_vectorstore(chuncks):
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GOOGLE_API_KEY"))
    vectorstore=FAISS.from_texts(texts=chuncks,embedding=embedding)
    return vectorstore

    
def get_component(text_chunks):
    prompt_template = """The following is a document
    {doc}
    Based on this doc, please identify the main product [main product] and component [list of components] with their descrption [description]
    Helpful Answer:"""

    prompt = PromptTemplate(
        input_variables=["doc"], template=prompt_template
    )
    # prompt = prompt_template.format(topic_from=topic_from, topic=research_topic)

    chain = LLMChain(llm=model, prompt=prompt)

    response = chain.run(doc=text_chunks)
    return response


def retreive_Component(component_with_desc):
    lines = component_with_desc.split("\n")

    # Find the index of the line starting with "**Components:**"
    components_index = lines.index("**Components:**")

    # Extract component names from lines following the "**Components:**" line
    component_names = [line.strip("* ").split(":")[0].strip() for line in lines[components_index + 1:] if line.startswith("*")]

    # Print component names
    return component_names

# def get_user_story(component):
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


    prompt = PromptTemplate(
        input_variables=["text"], template=prompt_template
    )
    # prompt = prompt_template.format(topic_from=topic_from, topic=research_topic)

    chain = LLMChain(llm=model, prompt=prompt)

    response = chain.run(text=component)
    return response
    # with st._main:
    #     print(response)
    #     st.write(response)

def get_user_story(component):
    response_schemas = [
    ResponseSchema(name="Title", description="name of the user story"),
    ResponseSchema(
        name="Summary",
        description="A brief, descriptive title for the user story that summarizes its purpose or goal.",
    ),
    ResponseSchema(
        name="Description",
        description="A detailed explanation of the user story, including what needs to be done and why it's important. This should provide enough context for the development team to understand the requirements and objectives.",
    ),
    ResponseSchema(
        name="Acceptance",
        description="Specific conditions or criteria that must be met for the user story to be considered complete. These criteria help define the scope of the work and provide a basis for testing.",
    ),
    ResponseSchema(
        name="Priority",
        description="Indicates the relative importance or urgency of the user story compared to other items in the backlog. This helps the team prioritize their work and focus on the most valuable tasks first.",
    ),
    ResponseSchema(
        name="Story Points",
        description="An estimate of the relative effort or complexity of the user story, often represented using a numerical value such as story points or ideal days. This helps the team understand the size of the task and plan their work accordingly.",
    ),
    ResponseSchema(
        name="Assignee",
        description="The team member responsible for implementing the user story. Assigning tasks to specific team members helps clarify accountability and ensures that work is evenly distributed.",
    ),
    ResponseSchema(
        name="Epic/Theme",
        description="If the user story is part of a larger initiative or project, it may be linked to an epic or theme in JIRA. This provides additional context and helps organize related work items.",
    ),
    ResponseSchema(
        name="Labels/Tags",
        description="Optional labels or tags that provide additional categorization or metadata for the user story. This can be useful for filtering and searching for related items in the backlog.",
    ),
    ResponseSchema(
        name="Linked Issues",
        description="Links to other issues or tasks that are related to the user story, such as dependencies or follow-up work. This helps ensure that the team is aware of any interconnected tasks or requirements.",
    ),
    ]

    sample_output="""
        "Login" : {
        "Title": "Login",\n
	    "Summary": "As a user, I want to be able to log in to the website to access my account and make purchases.",\n
	    "Description": "The login feature should allow users to enter their email address and password to access their account. If the user has forgotten their password, they should be able to reset it through a link sent to their email address.",\n
	    "Acceptance": "The user story will be considered complete when the following acceptance criteria are met:\n
        1. The user can successfully log in to the website using their email address and password.\n
        2. The user can reset their password if they have forgotten it.",\n
	    "Priority": "High",\n
	    "Story Points": "3",\n
	    "Assignee": "John Doe",\n
	    "Epic/Theme": "User Authentication",\n
	    "Labels/Tags": "login, authentication",\n
	    "Linked Issues": "None"
    }
    """
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt_template = """using following
    {text}
    Based on this text, create a user story for the buisnees for JIRA.
    When creating user stories in JIRA, it's important to include key attributes to provide clarity and context for the development team. Here are the key attributes typically used when creating user stories in JIRA:

    Note: Create user stories for each components from  this component listed {component_names}.\n This is the format of the user story {format_instructions} and do not include the word json and ''' on the output.
    
    Here is an example of the output:
    {sample_output}
    Helpful Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text","component_names"],
        partial_variables={"format_instructions": format_instructions},
    )

    user_stories_by_component = {}

    chain = LLMChain(llm=model, prompt=prompt)

    # Iterate over each component name
    for component_name in component:
        # Generate user story for the current component
        response_tt = chain.run(text=component, component_names=[component_name])
        # print(response_tt)
        # Add the generated user story to the dictionary under the component name key
        user_stories_by_component[component_name] = response_tt
    return user_stories_by_component

# def generate_pdf(user_stories):
    print(user_stories)
    data=json.loads(user_stories)
    # # Convert the dictionary to a DataFrame
    # df = pd.DataFrame(columns=["Title","Summary","Description","Acceptance","Priority","Story Points","Assignee","Epic/Theme","Labels/Tags","Linked Issuess"])
    
    # Transpose the DataFrame to have components as rows and attributes as columns
    # df = df.transpose()

    # # Specify the filename for the CSV file
    # csv_filename = "user_stories.csv"

    # # Save the DataFrame to CSV file
    # df.to_csv(csv_filename)


def generate_pdf(user_stories):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # print(user_stories)
    for i in user_stories.split("}"):
        print(i)
    # # Iterate over each component and its user story
    # for component, user_story in user_stories:
    #     pdf.cell(200, 10, txt=f"Component: {component}", ln=True, align="L")
    #     pdf.cell(200, 10, txt="", ln=True, align="L")  # Add an empty line for spacing
    #     # Add user story details
    #     for key, value in user_story:
    #         pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align="L")
    #     pdf.cell(200, 10, txt="", ln=True, align="L")  # Add an empty line between user stories

    # pdf_output_path = "user_stories.pdf"
    # pdf.output(pdf_output_path)
    # print(f"PDF generated successfully: {pdf_output_path}")

    # return pdf_output_path


def main():
    st.header("Upload your BRD")
    input_brd=st.file_uploader(
        "Upload your BRDs here and click Generate",accept_multiple_files=False,type="pdf")
    # get_chunk(input_brd)
    if st.button("Generate"):
        #Get the BRD text
        raw_text=get_brd_text(input_brd)
        #Get the chucks of the raw text from PDF
        text_chunks=get_text_chunk(raw_text)
        #Creating vector store
        vector_store= get_vectorstore(text_chunks)

        #Storing the collected compoenent from the BRD to the session state to access on other pages
        st.session_state["Components"]=get_component(text_chunks)

        component=retreive_Component(st.session_state["Components"])

        #Storing the generated user stories from the model to the session state to access on other pages
        user_stories=get_user_story(component)
        st.session_state["User Story"]=user_stories

        # for i in user_stories.values():
        #     print(i,"\n")
        #Generating the CSV File
        generate_pdf(user_stories)
