from dataclasses import dataclass
from typing import List


@dataclass
class RedditMemePost:
    """Representation of a collected Reddit meme post."""

    sample_id: str
    image_url: str
    title: str
    subreddit: str


TARGET_SUBREDDITS = [
    "dankmemes",
    "PoliticalHumor",
    "ComedyCemetery",
    "ControversialHumor",
]


def collect_reddit_image_posts(
    reddit,
    limit: int = 100,
) -> List[RedditMemePost]:
    """
    Collect image posts from the project-defined Reddit subreddits.

    The Reddit client is supplied by the caller so that authentication
    and credentials remain outside this module.

    Parameters
    ----------
    reddit:
        An authenticated PRAW Reddit client.

    limit:
        Maximum number of posts requested per subreddit.

    Returns
    -------
    List[RedditMemePost]
        Collected image posts with their titles used as meme context.
    """

    samples = []

    for subreddit_name in TARGET_SUBREDDITS:
        subreddit = reddit.subreddit(subreddit_name)

        for post in subreddit.hot(limit=limit):
            url = getattr(post, "url", "")

            if not url:
                continue

            if not _is_image_url(url):
                continue

            samples.append(
                RedditMemePost(
                    sample_id=str(post.id),
                    image_url=url,
                    title=str(post.title),
                    subreddit=subreddit_name,
                )
            )

    return samples


def _is_image_url(url: str) -> bool:
    """Return True when a URL points to a supported image format."""

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    )

    return url.lower().split("?")[0].endswith(image_extensions)