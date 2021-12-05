import os
import random
import re
import sys
import copy

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
    # Initialise model with one key for each page in corpus
    pagerank = {}
    for pages in corpus:
        pagerank[pages] = 1 / len(corpus)

    # If page doesn't have any links, interpret as one link for every page
    for pages in corpus:
        if not corpus[pages]:
            corpus[pages] = set(pages for pages in corpus)

    # Create dictionary of which pages link to each page
    links_to_page = crawl_origin(corpus)

    # Calculate page rank repeatedly
    while True:
        pagerank_working = copy.deepcopy(pagerank)

        for pages in corpus:
            first_sum = (1 - damping_factor) / len(corpus)
            second_sum = 0
            # Loop over all pages that link to current page
            for origin in links_to_page[pages]:
                second_sum += pagerank[origin] / len(corpus[origin])
            pagerank_working[pages] = first_sum + (damping_factor * second_sum)

        # Check whether iterations have converged
        max_diff = 0
        for page in pagerank_working:
            if abs(pagerank[page] - pagerank_working[page]) > max_diff:
                max_diff = abs(pagerank[page] - pagerank_working[page])
        # Stop loop when no value changes by more than 0.001
        if max_diff <= 0.001:
            break
        pagerank = pagerank_working

    return pagerank

def crawl_origin(corpus):
    """
    Parse a directory of HTML pages and check for links from other pages.
    Return a dictionary where each key is a page, and values are
    a set of all pages that link TO that page.
    """

    # Initialise with blank set
    links_to_page = {}
    for page in corpus:
        links_to_page[page] = set()

    # Add in actual links
    for page in corpus:
        for link in corpus[page]:
            links_to_page[link].add(page)
    return links_to_page

def update_corpus(corpus):
    """
    For use with iterate_pagerank function.  Fulfils requirement that a page 
    that has no links at all should be interpreted as having one link for every 
    page in the corpus (including itself).
    """

    for page in corpus:
        if not corpus[page]:
            corpus[page] = set(page for page in corpus)

    return corpus

if __name__ == "__main__":
    main()
