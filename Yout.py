import os
import time
import json
import asyncio
import concurrent.futures
import isodate
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK resource (using the standard 'punkt')
nltk.download('punkt')

# Optional: Use environment variable for API key (recommended for production)
API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_API_KEY_GOES_HERE")
youtube = build("youtube", "v3", developerKey=API_KEY)

# Transformer-based summarizer for accurate summaries
try:
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except ImportError:
    summarizer = None
    print("Transformers library not installed. Falling back to heuristic summarization.")


# Function to search YouTube videos
def search_videos(query, max_results=10):
    request = youtube.search().list(
        q=query, part="id,snippet", type="video", maxResults=max_results
    )
    response = request.execute()
    videos = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        videos.append({
            "video_id": video_id,
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={video_id}"
        })
    return videos

# Function to fetch video details in batch
def get_video_details(video_ids):
    request = youtube.videos().list(
        part="statistics,contentDetails",
        id=",".join(video_ids)
    )
    response = request.execute()
    video_details = {}
    for item in response.get("items", []):
        video_id = item["id"]
        stats = item.get("statistics", {})
        details = item.get("contentDetails", {})
        # Convert ISO 8601 duration to minutes
        try:
            duration_seconds = isodate.parse_duration(details["duration"]).total_seconds()
        except Exception:
            duration_seconds = 0  
        video_details[video_id] = {
            "video_id": video_id,
            "like_count": int(stats.get("likeCount", 0)),
            "view_count": int(stats.get("viewCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
            "duration_minutes": round(duration_seconds / 60, 2)
        }
    return video_details

# Asynchronous transcript fetching with retry logic
async def fetch_transcript_async(video_id, retries=3, delay=1):
    for attempt in range(retries):
        try:
            transcript = await asyncio.to_thread(YouTubeTranscriptApi.get_transcript, video_id)
            transcript_text = " ".join(entry["text"] for entry in transcript)
            return video_id, transcript_text
        except Exception as e:
            if attempt == retries - 1:
                return video_id, None
            await asyncio.sleep(delay)

async def get_video_transcripts_async(video_ids):
    tasks = [fetch_transcript_async(vid) for vid in video_ids]
    results = await asyncio.gather(*tasks)
    transcripts = {video_id: transcript_text for video_id, transcript_text in results}
    return transcripts

# Advanced summarization function: uses a transformer-based summarizer if available
def summarize_transcript(text):
    if not text:
        return "Summary not available."
    # If text is too long for the model, split into manageable chunks.
    if summarizer:
        try:
            # The model can have max token limitations; here we use a simple split strategy.
            max_chunk_size = 1024
            text_chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                         for chunk in text_chunks]
            # Combine summaries and run one more summarization pass for coherence.
            combined_summary = " ".join(summaries)
            final_summary = summarizer(combined_summary, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            return final_summary
        except Exception as e:
            # Fall back to heuristic if transformer fails
            pass
    # Fallback: simple heuristic summarization
    sentences = np.array(sent_tokenize(text))
    return " ".join(sentences[:5])

# Main execution function
def main():
    start_time = time.time()
    
    # Step 1: Search Videos
    query = "Motion in 2D"
    jee_videos = search_videos(query, max_results=10)
    video_ids = [vid["video_id"] for vid in jee_videos]
    
    # Step 2: Fetch Video Details in Batch
    video_details = get_video_details(video_ids)
    
    # Step 3: Filter & Rank Videos
    filtered_videos = []
    for vid in jee_videos:
        details = video_details.get(vid["video_id"])
        if details and details["duration_minutes"] >= 3 and details["view_count"] > 1000:
            # A more robust scoring mechanism might include normalization
            importance_score = (details["view_count"] * 0.7) + (details["like_count"] * 0.2) + (details["comment_count"] * 0.1)
            vid.update(details)
            vid["importance_score"] = importance_score
            filtered_videos.append(vid)
    
    filtered_videos.sort(key=lambda x: x["importance_score"], reverse=True)
    
    # Step 4: Asynchronously Fetch Transcripts for Top Videos
    top_videos = filtered_videos[:5]
    top_video_ids = [vid["video_id"] for vid in top_videos]
    transcripts = asyncio.run(get_video_transcripts_async(top_video_ids))
    
    # Step 5: Summarize Transcripts using advanced summarization
    for vid in top_videos:
        transcript_text = transcripts.get(vid["video_id"])
        vid["summary"] = summarize_transcript(transcript_text)
    
    # Step 6: Save Results to JSON and CSV
    with open("jee_videos.json", "w", encoding="utf-8") as json_file:
        json.dump(top_videos, json_file, indent=4, ensure_ascii=False)
    
    df = pd.DataFrame(top_videos)
    df.to_csv("jee_videos.csv", index=False)
    
    # Print Results
    for vid in top_videos:
        print(f"📌 {vid['title']} ({vid['duration_minutes']} min)")
        print(f"👍 {vid['like_count']} Likes | 👀 {vid['view_count']} Views | 💬 {vid['comment_count']} Comments")
        print(f"🔗 {vid['url']}")
        print(f"📝 Summary: {vid['summary']}\n")
    
    print(f"✅ Process completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
