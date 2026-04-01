import re
import pdfplumber
from docx import Document
import spacy
import os


SPACY_MODEL_NAME = "en_core_web_lg" #"en_core_web_sm"
try:
    nlp = spacy.load(SPACY_MODEL_NAME)
except OSError:
    print(f"Spacy model '{SPACY_MODEL_NAME}' not found. Downloading...")
    os.system(f"python -m spacy download {SPACY_MODEL_NAME}")
    nlp = spacy.load(SPACY_MODEL_NAME)

class ResumePreprocessor:
    def __init__(self):
        self.section_patterns = {
            'header': r'^(summary|profile|objective)', 
            'experience': r'^(work\s*experience|professional\s*experience|experience)',
            'education': r'^(education|academic\s*background)',
            'skills': r'^(technical\s*skills|key\s*skills|skills)',
            'projects': r'^(projects|personal\s*projects)',
        }

    def convert_to_text(self, file_path):
        text = ""
        try:
            if file_path.lower().endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    text = '\n'.join(page.extract_text() or '' for page in pdf.pages) 
            elif file_path.lower().endswith('.docx'):
                doc = Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            elif file_path.lower().endswith('.txt'): 
                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: 
                    text = f.read()
            else:
                print(f"Warning: Unsupported file type: {file_path}. Attempting to read as text.")
                try: 
                     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except Exception as e_txt:
                    print(f"Could not read file {file_path} as text: {e_txt}")
                    return "" 

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return ""
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return "" 
        return text

    def clean_text(self, text):
        if not text: 
            return ""
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        
        text = re.sub(r'[^\w\s.,\-@+_/()]', '', text)
        
        
        return text

    def segment_resume(self, text):
        
        sections = {'header': []} 
        current_section = 'header'
        lines = text.split('\n') 

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            matched_section = None
            for section, pattern in self.section_patterns.items():
                
                if re.match(pattern, line_clean, re.IGNORECASE):
                    matched_section = section
                    break

            if matched_section and matched_section != current_section:
                 # Found a new section header
                current_section = matched_section
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(line_clean) 
            elif current_section in sections:
                
                sections[current_section].append(line) 
            
            

        
        return {k: '\n'.join(v).strip() for k, v in sections.items() if v} 
