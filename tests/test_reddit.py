from types import SimpleNamespace

from src.multimodal_hate.data.scraping.reddit import (
    collect_reddit_image_posts,
)


class FakeSubreddit:
    def __init__(self, name):
        self.name = name

    def hot(self, limit):
        return [
            SimpleNamespace(
                id="reddit123",
                url="https://example.com/meme.jpg",
                title="Example meme title",
            ),
            SimpleNamespace(
                id="reddit456",
                url="https://example.com/not-image.html",
                title="Not an image",
            ),
        ]


class FakeReddit:
    def subreddit(self, name):
        return FakeSubreddit(name)


def test_collect_reddit_image_posts():
    reddit = FakeReddit()

    samples = collect_reddit_image_posts(
        reddit,
        limit=2,
    )

    assert len(samples) == 4

    sample = samples[0]

    assert sample.sample_id == "reddit123"
    assert sample.image_url == "https://example.com/meme.jpg"
    assert sample.title == "Example meme title"
    assert sample.subreddit == "dankmemes"