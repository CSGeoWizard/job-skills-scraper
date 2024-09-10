import os
import csv
import pandas as pd
from jobspy import scrape_jobs
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re
from openai import OpenAI

class JobAnalyzer:
    def __init__(
        self,
        search_term,
        site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor"],
        location="USA",
        results_wanted=20,
        api_key=None
    ):
        self.site_name = site_name
        self.search_term = search_term
        self.location = location
        self.results_wanted = results_wanted
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.df = None
        self.summary = {}

        self.cleaned_search_term = re.sub(r'[\'\"\\\/]', '', self.search_term)

    def scrape_jobs(self):
        jobs = scrape_jobs(
            site_name=self.site_name,
            search_term=self.search_term,
            location=self.location,
            results_wanted=self.results_wanted,
            country_indeed='USA'
        )
        self.df = pd.DataFrame(jobs)
        self.summary['Total Jobs Found'] = len(jobs)

    def adjust_salary(self, salary):
        if pd.notna(salary) and salary > 0:
            if salary < 1000:
                return salary * 1000
        return salary

    def calculate_salaries(self):
        self.df['min_amount'] = self.df['min_amount'].apply(self.adjust_salary)
        self.df['max_amount'] = self.df['max_amount'].apply(self.adjust_salary)

        avg_low_salary = self.df['min_amount'].mean()
        avg_high_salary = self.df['max_amount'].mean()

        self.summary['Average Low Salary'] = avg_low_salary
        self.summary['Average High Salary'] = avg_high_salary

    def calculate_remote_percentage(self):
        self.df['is_remote'] = self.df['is_remote'].astype(bool)
        remote_percentage = self.df['is_remote'].mean() * 100
        self.summary['Remote Job Percentage'] = remote_percentage

    def extract_technical_skills(self, description):
        # Define prompt for required skills
        required_prompt = f"Extract the hard technical skills from the 'Required Education, Experience, & Skills' section of the following job description. Only include specific tools, techniques, and processes (omit mentions of experience or soft skills):\n\n{description}\n\nList only the hard technical skills."

        # Define prompt for desired skills
        desired_prompt = f"Extract the hard technical skills from the 'Preferred Education, Experience, & Skills' section of the following job description. Only include specific tools, techniques, and processes (omit mentions of experience or soft skills):\n\n{description}\n\nList only the hard technical skills."

        # Call GPT for required skills
        required_completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that extracts hard technical skills from job descriptions."},
                {"role": "user", "content": required_prompt}
            ]
        )
        
        # Call GPT for desired skills
        desired_completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that extracts hard technical skills from job descriptions."},
                {"role": "user", "content": desired_prompt}
            ]
        )

        # Get the responses
        required_skills = required_completion.choices[0].message.content.strip()
        desired_skills = desired_completion.choices[0].message.content.strip()

        # Save the prompts and responses for review
        return {
            "required": required_skills, 
            "desired": desired_skills, 
            "required_prompt": required_prompt, 
            "desired_prompt": desired_prompt
        }

    def analyze_technical_skills(self):
        # Extract both required and desired technical skills
        skills = self.df['description'].dropna().apply(self.extract_technical_skills)
        self.df['required_skills'] = skills.apply(lambda x: x['required'])
        self.df['desired_skills'] = skills.apply(lambda x: x['desired'])

        # Optionally, store the prompts in the dataframe for future review
        self.df['required_prompt'] = skills.apply(lambda x: x['required_prompt'])
        self.df['desired_prompt'] = skills.apply(lambda x: x['desired_prompt'])

    nltk_data_path = os.path.join(os.getenv("HOME"), "nltk_data")

    def ensure_nltk_stopwords(self):
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', download_dir=nltk_data_path)
            stop_words = set(stopwords.words('english'))
        stop_words.add('experience')  # Add the word "experience" to stopwords
        return stop_words

    def clean_text(self, text):
        stop_words = self.ensure_nltk_stopwords()

        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove stopwords, including "experience"
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    def create_wordcloud(self, skills_type):
        column = f'{skills_type}_skills'
        cleaned_text = self.df[column].dropna().apply(self.clean_text)
        joined_text = ' '.join(cleaned_text)

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(joined_text)

        # Plot the WordCloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # Save the word cloud as a PNG file
        wordcloud_path = f'{self.cleaned_search_term}_{skills_type}_wordcloud.png'
        plt.savefig(wordcloud_path)
        plt.show()
        print(f"Word cloud for {skills_type} skills saved as {wordcloud_path}")

    def create_histogram(self, skills_type):
        column = f'{skills_type}_skills'
        cleaned_text = self.df[column].dropna().apply(self.clean_text)
        joined_text = ' '.join(cleaned_text)

        # Count word occurrences using Counter
        word_counts = Counter(joined_text.split())

        # Get the top 25 most common words
        most_common_words = word_counts.most_common(25)

        # Separate words and counts for plotting
        words, counts = zip(*most_common_words)

        # Plot a histogram for the top 25 words
        plt.figure(figsize=(12, 6))
        plt.bar(words, counts, color='skyblue')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title(f'Top 25 Most Frequent Words in {skills_type} Skills')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the histogram as a PNG file
        histogram_path = f'{self.cleaned_search_term}_{skills_type}_histogram.png'
        plt.savefig(histogram_path)
        plt.show()
        print(f"Histogram for {skills_type} skills saved as {histogram_path}")


    def save_summary_to_csv(self):
        summary_df = pd.DataFrame([self.summary])
        summary_file = f'{self.cleaned_search_term}_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary statistics saved to '{summary_file}'")

    def save_jobs_to_csv(self):
        # Save job data including prompts and GPT responses
        jobs_file = f'{self.cleaned_search_term}_jobs.csv'
        self.df.to_csv(jobs_file, index=False)
        print(f"Job data saved to '{jobs_file}'")

    def run(self):
        self.scrape_jobs()
        self.calculate_salaries()
        self.calculate_remote_percentage()
        self.analyze_technical_skills()
        self.create_wordcloud('required')  # Word cloud for required skills
        self.create_wordcloud('desired')  # Word cloud for desired skills
        self.create_histogram('required')  # Histogram for required skills
        self.create_histogram('desired')  # Histogram for desired skills
        self.save_summary_to_csv()
        self.save_jobs_to_csv()
