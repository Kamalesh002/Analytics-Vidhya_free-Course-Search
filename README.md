# Analytics Vidhya Course Search

This is a Streamlit application that allows users to search and filter free courses available on Analytics Vidhya. The app scrapes the course data from the website and enables searching by keyword, filtering by category, and sorting by various criteria such as Rating or Reviews Count. It uses Sentence Transformers to calculate course similarity scores to help find the most relevant results based on a search query.

## Features

- Scrapes course data from Analytics Vidhya's free courses page.
- Search courses by keywords (e.g., "Machine Learning for Beginners").
- Filter courses by categories (e.g., Data Science, Python).
- Sort results by Relevance, Rating, or Reviews Count.
- Displays course information including title, price, lesson count, ratings, and reviews.
- Built with Streamlit for an interactive web interface.

## How It Works

- The app scrapes course data from the Analytics Vidhya Free Courses page.
- It uses BeautifulSoup to parse the HTML content and extract course details such as title, categories, lesson count, price, and rating.
- The SentenceTransformer model is used to generate embeddings of course descriptions for similarity-based search.
- The app calculates the cosine similarity between the user's query and course descriptions to rank search results by relevance.
