import feedparser

def run(text):
    url = "http://feeds.bbci.co.uk/news/rss.xml"
    feed = feedparser.parse(url)
    if feed.entries:
        headlines = [entry.title for entry in feed.entries[:3]]
        return "📰 Top News:\n   • " + "\n   • ".join(headlines)
    return "❌ Unable to fetch news."