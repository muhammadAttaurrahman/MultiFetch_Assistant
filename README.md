# MultiFetch_Assistant
Our "MultiFetch Assistant" is a multipurpose LLM application, suitable for ordinary users. Our app can interact with PDF documents, compare resumes to job descriptions, extract content from photos, and summarize transcripts of YouTube videos using LLM.

We have used the following packages:

* Google's Generative AI(Gemini): Google's advanced AI models are made to generate text, graphics, and other types of content.

* Longchain: A framework for creating language model-powered applications, including important functionalities like text splitting, text embedding, and language model integration.

* Streamlit: An open-source application framework for building and sharing custom web applications.

* YouTube_Transcript_API: A Python package for getting a YouTube video's transcript.

* Pdf2image: A Python package that creates image formats from PDF files.

* PyPDF2: A library for reading and manipulating PDF files.


Run:

Make a ".env" file and store your Google Gemini API Key in the following way in this file:

GOOGLE_API_KEY = " Your API KEY"

!pip install -r requirement.txt

to run the application, use -> "streamlit run multiFetch_Assistant.py" in the terminal.
