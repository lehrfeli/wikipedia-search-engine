# Scrapy spider that crawls Wikipedia starting from a seed URL (e.g. the
# "Law" article), following in-article links and saving each page's url,
# title, and body text as one JSON line per page.
#
# Not run directly with `scrapy crawl` — it's launched with `scrapy runspider`
# and command-line arguments via crawler.bat (or manually, see README):
#   scrapy runspider wiki_spider.py -a seed_file=seed.txt -a num_pages=100000 \
#       -a hops=6 -a output_dir=../data/crawled

import scrapy
import os

class WikiSpider(scrapy.Spider):
    name = "wiki_spider"

    # --- setup: read CLI args passed in via `scrapy runspider -a ...` ---
    def __init__(self, seed_file, num_pages, hops, output_dir, *args, **kwargs):
        super(WikiSpider, self).__init__(*args, **kwargs)

        # Opens the seed file from the command line argument to get the start URL
        with open(seed_file, 'r') as f:
            self.start_urls = [line.strip() for line in f if line.strip()]

        self.num_pages = num_pages
        # We accept 'hops' to satisfy the script input, but we won't use it to limit the crawl

        # Sets up the output directory path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_path = os.path.join(output_dir, 'wiki_data.jsonl')

    # --- scrapy crawl settings, built from the CLI args above ---
    @property # This allows the values typed into the terminal to communicate with crawler code
    def custom_settings(self):
        return {
            'CLOSESPIDER_PAGECOUNT': int(self.num_pages), # stops running after reaching pagecount value
            'CONCURRENT_REQUESTS': 16, # allows the spider to download 16 pages at the same time. This is standard and keeps wiki from blocking
            'FEEDS': {self.output_path: {'format': 'jsonlines', 'overwrite': True}}, # handles the file saving for so i don't have to write open() or write() in the code
            'ROBOTSTXT_OBEY': False, # ignores wiki's rules for bots scraping
            # prioritize shallow links to keep the content as relevant to 'law' as possible
            'DEPTH_PRIORITY': 1,
            # force first-in-first-out queue
            'SCHEDULER_DISK_QUEUE': 'scrapy.squeues.PickleFifoDiskQueue',
            'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue',
        }

    # --- page parsing: extract title/text, save the page, follow links ---
    def parse(self, response):
        # get title
        # 'string()' operates like a function and grabs text even if it's trapped inside a <span> tag. 
        # Ttakes the H1 element and converts the entire tree structure into a single flat string.
        # uses universal xpath formula //tag[@attribute=value] so that its applcable to all titles
        # html attribute and value for title is id="firstHeading". Use inspect on wiki page to get this
        title = response.xpath('string(//h1[@id="firstHeading"])').get()

        # get text data
        # Grab every <p>, convert to string, join them, and clean whitespace
        # div#mw-content-text is the name for the main body of the page
        raw_text = [p.xpath('string(.)').get() for p in response.css('div#mw-content-text p')]
        text_content = " ".join(" ".join(raw_text).split())
        
        # save and set limit to wiki pages with over 500 words
        if title and len(text_content) > 500:
            yield {
                'url': response.url,
                'title': title,
                'text': text_content
            }

        # Follow Links
        # CSS telling crawler to look inside the body of the wiki page (div#mw-content-text this can be
        # found using inspector on the wiki page). this looks for <div>, then identifies id mw-content-text
        # a::attr(href) tells it to find the url in the body and go there
        for link in response.css('div#mw-content-text a::attr(href)').getall():
            if link.startswith('/wiki/') and ':' not in link:
                yield response.follow(link)


