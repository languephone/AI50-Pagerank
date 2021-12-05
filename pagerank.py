import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    model = {}
    link_count = len(corpus[page])

    # Initialise model with one key for each page in corpus
    for pages in corpus:
        model[pages] = (1 - damping_factor) / len(corpus)

    # If no links on page, return even distribution among all pages in corpus
    if len(corpus[page]) < 1:
        for pages in corpus:
            model[pages] += damping_factor / len(corpus)
    else:
        # Add portion of damping factor to each link
        for link in corpus[page]:
            model[link] += damping_factor / link_count

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialise model with one key for each page in corpus
    pagerank = {}
    for pages in corpus:
        pagerank[pages] = 0

    # Generate staring page at random
    sample_start = random.choice(list(corpus))

    # Generate remaining samples based on transition model
    for i in range(n):
        next_page_model = transition_model(corpus, sample_start, damping_factor)

        # Convert transition model to list for use in random.choices function
        weighting = list(next_page_model.values())

        # Select next page based on transition model options & weighting
        next_page = random.choices(list(next_page_model), weights=weighting)[0]
        pagerank[next_page] += 1

        # Redefine start page for next loop
        sample_start = next_page

    # Convert raw count of page views to probability distribution
    for page in pagerank:
        pagerank[page] = pagerank[page] / n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
