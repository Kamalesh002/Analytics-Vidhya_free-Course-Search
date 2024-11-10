import subprocess
import sys
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

# Install and import beautifulsoup4
try:
    from bs4 import BeautifulSoup
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
    from bs4 import BeautifulSoup

# Install and import requests
try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

# Install and import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

# Install and import scikit-learn
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.metrics.pairwise import cosine_similarity



class AnalyticsVidhyaCourseSearch:
    def __init__(self):
        self.courses_data = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.last_updated = None

    def scrape_courses(self, url: str) -> int:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            course_containers = soup.find_all('li', class_='course-cards__list-item')

            self.courses_data = []  # Reset existing data

            for course in course_containers:
                try:
                    # Enhanced course detail extraction with error handling
                    title = course.find('h3').text.strip() if course.find('h3') else "No title available"
                    categories = course.find('h4').text.strip() if course.find('h4') else "No categories available"
                    lesson_count = course.find('span', class_='course-card__lesson-count').text.strip() if course.find('span', class_='course-card__lesson-count') else "N/A"
                    price = course.find('span', class_='course-card__price').text.strip() if course.find('span', class_='course-card__price') else "N/A"

                    course_link = course.find('a', class_='course-card')['href'] if course.find('a', class_='course-card') else None
                    image_url = course.find('img', class_='course-card__img')['src'] if course.find('img', class_='course-card__img') else None

                    rating_container = course.find('div', class_='course-card__reviews')
                    if rating_container:
                        stars_count = len(rating_container.find_all('i', class_='fa-star'))
                        reviews_count = rating_container.find('span', class_='review__stars-count').text.strip('()') if rating_container.find('span', class_='review__stars-count') else "0"
                    else:
                        stars_count = 0
                        reviews_count = "0"

                    course_data = {
                        'title': title,
                        'categories': categories,
                        'lesson_count': lesson_count,
                        'price': price,
                        'course_link': f"https://courses.analyticsvidhya.com{course_link}" if course_link else None,
                        'image_url': image_url,
                        'rating': stars_count,
                        'reviews_count': reviews_count,
                        'combined_text': f"{title} {categories} {lesson_count}"
                    }
                    self.courses_data.append(course_data)

                except AttributeError as e:
                    st.warning(f"Error processing a course: {str(e)}")
                    continue

            self._generate_embeddings()
            self.last_updated = datetime.now()
            return len(self.courses_data)

        except requests.RequestException as e:
            st.error(f"Error fetching the webpage: {e}")
            return 0

    def _generate_embeddings(self):
        # Generate embeddings for the courses data
        texts = [course['combined_text'] for course in self.courses_data]
        self.embeddings = self.model.encode(texts)

    def search_courses(self, query: str, num_results: int):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        results = []
        for idx, similarity in enumerate(similarities):
            course = self.courses_data[idx].copy()
            course['similarity_score'] = similarity
            results.append(course)
        
        # Sort by similarity and limit results
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:num_results]
        return results

# Streamlit UI Enhancements
st.set_page_config(
    page_title="Analytics Vidhya Course Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .course-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: white;
        color: #333; /* Darker color for general text */
    }
    .course-card h3 {
        color: #1a73e8; /* Change heading color to a visible shade, e.g., blue */
    }
    .rating-stars {
        color: gold;
    }
    .course-image {
        border-radius: 5px;
        width: 100%;
        max-width: 300px;
    }
    .last-updated {
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'search_system' not in st.session_state:
    st.session_state.search_system = AnalyticsVidhyaCourseSearch()
    st.session_state.courses_loaded = False
    st.session_state.filter_categories = []

# Sidebar enhancements
with st.sidebar:
    st.header("Data Management")
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Load/Refresh Courses", type="primary"):
            with st.spinner("Loading courses..."):
                url = "https://courses.analyticsvidhya.com/pages/all-free-courses"
                num_courses = st.session_state.search_system.scrape_courses(url)
                if num_courses > 0:
                    st.session_state.courses_loaded = True
                    st.success(f"Successfully loaded {num_courses} courses!")
                else:
                    st.error("Failed to load courses. Please try again.")
    
    if st.session_state.courses_loaded:
        st.markdown("### Filters")
        # Extract unique categories
        all_categories = set()
        for course in st.session_state.search_system.courses_data:
            categories = [cat.strip() for cat in course['categories'].split(',')]
            all_categories.update(categories)
        
        st.session_state.filter_categories = st.multiselect(
            "Filter by Category:",
            sorted(list(all_categories))
        )

# Main interface enhancements
st.title("📚 Analytics Vidhya Course Search")

if st.session_state.search_system.last_updated:
    st.markdown(f"<p class='last-updated'>Last updated: {st.session_state.search_system.last_updated.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

# Search interface
if st.session_state.courses_loaded:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("🔍 Search courses:", placeholder="e.g., machine learning for beginners")
    with col2:
        num_results = st.number_input("Number of results:", min_value=1, max_value=10, value=5)
    with col3:
        sort_by = st.selectbox("Sort by:", ["Relevance", "Rating", "Reviews Count"])

    if query:
        results = st.session_state.search_system.search_courses(query, num_results)
        
        # Apply category filters
        if st.session_state.filter_categories:
            results = [r for r in results if any(cat.strip() in st.session_state.filter_categories 
                                               for cat in r['categories'].split(','))]
        
        # Sort results
        if sort_by == "Rating":
            results.sort(key=lambda x: x['rating'], reverse=True)
        elif sort_by == "Reviews Count":
            results.sort(key=lambda x: int(x['reviews_count']), reverse=True)

        # Display results
        if results:
            for i, result in enumerate(results, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="course-card">
                        <h3>{i}. {result['title']}</h3>
                        <p><strong>Categories:</strong> {result['categories']}</p>
                        <p><strong>Rating:</strong> {'⭐' * int(result['rating'])} ({result['reviews_count']} reviews)</p>
                        <p><strong>Details:</strong> {result['lesson_count']} | {result['price']}</p>
                        <p><strong>Similarity Score:</strong> {result['similarity_score']:.2f}</p>
                        <a href="{result['course_link']}" target="_blank">View Course</a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No results found. Try a different search query or adjust your filters!")
else:
    st.warning("⚠️ Please load the courses first using the button in the sidebar!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created with ❤️ using Streamlit and Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)
