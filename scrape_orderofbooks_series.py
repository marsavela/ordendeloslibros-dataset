# Scraper for orderofbooks.com: Extracts series, author, and (if available) books, outputs as JSON
import requests
from bs4 import BeautifulSoup
import json
import re
import time
from typing import Optional

def get_author_links_from_sitemap():
    """Try to get author links from sitemap or robots.txt"""
    base_url = "https://www.orderofbooks.com"
    author_links = []
    
    # Try to find sitemap
    for sitemap_path in ['/sitemap.xml', '/sitemap_index.xml', '/sitemap-authors.xml']:
        try:
            resp = requests.get(base_url + sitemap_path)
            if resp.status_code == 200:
                # Look for author URLs in sitemap
                author_urls = re.findall(r'<loc>(https://www\.orderofbooks\.com/authors/[^<]+)</loc>', resp.text)
                author_links.extend(author_urls)
                print(f"Found {len(author_urls)} author links from {sitemap_path}")
        except Exception as e:
            print(f"Could not access {sitemap_path}: {e}")
    
    return list(set(author_links))  # Remove duplicates

def get_author_links_from_pagination():
    """Get author links by following pagination on the authors page"""
    author_links = []
    base_url = "https://www.orderofbooks.com/authors/"
    
    # The authors page has pagination - try first few pages
    for page in range(1, 6):  # Test first 5 pages
        try:
            url = f"{base_url}?page={page}" if page > 1 else base_url
            print(f"Checking page {page}: {url}")
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Look for author links in the page content
            links = soup.find_all('a', href=re.compile(r'/authors/[^/]+/$'))
            page_links = []
            for link in links:
                href = link.get('href')
                if href and not href.endswith('/authors/'):  # Exclude the main authors page
                    full_url = f"https://www.orderofbooks.com{href}" if href.startswith('/') else href
                    page_links.append(full_url)
            
            print(f"  Found {len(page_links)} author links on page {page}")
            author_links.extend(page_links)
            
            if not page_links:  # No more links found, probably reached end
                break
                
            time.sleep(1)  # Be respectful
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
    
    return list(set(author_links))  # Remove duplicates

def get_sample_author_links():
    """Get a few sample author links for testing"""
    sample_authors = [
        "https://www.orderofbooks.com/authors/lee-child/",
        "https://www.orderofbooks.com/authors/james-patterson/",
        "https://www.orderofbooks.com/authors/stephen-king/",
        "https://www.orderofbooks.com/authors/agatha-christie/",
        "https://www.orderofbooks.com/authors/john-grisham/"
    ]
    return sample_authors

def get_author_links():
    """This function is no longer used but kept for reference"""
    pass

def scrape_author_series(author_url):
    """Scrape series and books for a given author using techniques from the main scraper."""
    print(f"Scraping author: {author_url}")
    try:
        resp = requests.get(author_url, timeout=10)
        if resp.status_code != 200:
            print(f"  Error: Status {resp.status_code}")
            return []
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find author name from the main heading
        author_name = None
        main_heading = soup.find('h1')
        if main_heading:
            # Extract author name from "Order of [Author Name] Books"
            heading_text = main_heading.text.strip()
            match = re.search(r'Order of (.+?) Books', heading_text)
            if match:
                author_name = match.group(1)
        
        if not author_name:
            print(f"  Could not find author name")
            return []
        
        print(f"  Found author: {author_name}")
        
        series_data = []
        
        # Method 1: Look for "ribbon" blocks (newer layout)
        ribbon_blocks = find_ribbon_blocks(soup)
        if ribbon_blocks:
            print(f"    Found {len(ribbon_blocks)} ribbon blocks")
            for series_name, books in ribbon_blocks:
                if books:
                    clean_name = series_name or f"{author_name} Books"
                    series_data.append({
                        "series_name": clean_name,
                        "author": author_name,
                        "books": books
                    })
                    print(f"      Added series '{clean_name}' with {len(books)} books")
        
        # Method 2: Look for heading + list blocks (older layout)
        if not series_data:
            print("    No ribbon blocks found, trying heading + list blocks...")
            heading_blocks = find_heading_list_blocks(soup)
            if heading_blocks:
                print(f"    Found {len(heading_blocks)} heading blocks")
                for series_name, books in heading_blocks:
                    if books:
                        series_data.append({
                            "series_name": series_name,
                            "author": author_name,
                            "books": books
                        })
                        print(f"      Added series '{series_name}' with {len(books)} books")
        
        print(f"  Total series found: {len(series_data)}")
        return series_data
        
    except Exception as e:
        print(f"  Error scraping {author_url}: {e}")
        return []


def clean_text(s: str) -> str:
    """Clean and normalize text"""
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_year(text: str) -> Optional[int]:
    """Extract year from text"""
    m = re.search(r"(19|20)\d{2}", text)
    return int(m.group(0)) if m else None


def normalize_series_name(heading_text: str) -> str:
    """Normalize series name from heading text"""
    t = clean_text(heading_text)
    t = re.sub(r"(?i)\b(books?|novels?|series)\b\s*in\s*order.*$", "", t)
    t = re.sub(r"(?i)\breading\s*order.*$", "", t)
    t = t.strip(" :–-—")
    return t or heading_text.strip()


def book_from_text(txt: str) -> Optional[dict]:
    """Create a book dict from raw list/text entry"""
    txt = clean_text(txt)
    # Skip JS expansion placeholder lines
    if re.search(r"(?i)show all books in this series", txt):
        return None
    # Remove leading index markers like "1." or "12)" etc.
    txt = re.sub(r"^\s*\d+[\.!\)]\s*", "", txt)
    year = extract_year(txt)
    title = re.sub(r"\((19|20)\d{2}\)", "", txt).strip(" -–—")
    return {"title": title, "year": year}


def parse_table_block(tbl):
    """Parse a ribbon/list table layout"""
    visibles = []
    hiddens = []
    for tr in tbl.select("tr"):
        if tr.find(class_="showall"):
            continue
        classes = tr.get("class", []) or []
        tt = tr.select_one("td.booktitle")
        yy = tr.select_one("td.bookyear")
        if not tt:
            tds = tr.find_all("td")
            if not tds:
                continue
            if len(tds) == 1 and re.search(r"(?i)show all books in this series", tds[0].get_text(" ", strip=True)):
                continue
            tt = tds[0]
            yy = tds[1] if len(tds) > 1 else None
        raw_title = clean_text(tt.get_text(" ", strip=True))
        # Remove any trailing expansion hint
        raw_title = re.sub(r"\s*\+\s*Show All Books in this Series\s*$", "", raw_title, flags=re.IGNORECASE)
        if re.search(r"(?i)show all books in this series", raw_title):
            continue
        if not raw_title:
            continue
        year = extract_year(yy.get_text(" ", strip=True)) if yy else None
        book = {"title": raw_title, "year": year}
        (hiddens if any(c == "hiderow" for c in classes) else visibles).append(book)
    return visibles + hiddens


def find_ribbon_blocks(soup):
    """Find ribbon blocks (newer layout)"""
    out = []
    for ribbon in soup.select("div.ribbon"):
        series_name = None
        cand = ribbon.find_previous(lambda t: t.name in ("h2", "h3", "h4"))
        while cand:
            raw_txt = cand.get_text(" ", strip=True)
            norm = normalize_series_name(raw_txt)
            if not re.match(r"^\(with\b", norm, re.IGNORECASE) and not norm.startswith("("):
                series_name = norm
                break
            cand = cand.find_previous(lambda t: t.name in ("h2", "h3", "h4"))
        if not series_name and cand:
            series_name = norm
        lst = ribbon.find_next_sibling("div", class_="list") or ribbon.find_next("div", class_="list")
        if not lst:
            continue
        tbl = lst.find("table")
        if not tbl:
            ulol = lst.find(lambda t: t.name in ("ul", "ol"))
            raw_books = [book_from_text(li.get_text(" ", strip=True)) for li in ulol.find_all("li", recursive=False)] if ulol else []
            books = [b for b in raw_books if b]
        else:
            books = parse_table_block(tbl)
        if books:
            out.append((series_name, books))
    return out


def find_heading_list_blocks(soup):
    """Find heading + list blocks (older layout)"""
    results = []
    for h in soup.select("h2, h3, h4"):
        htxt = clean_text(h.get_text(" ", strip=True))
        if not re.search(r"(?i)(book series in order|series in order|books in order|reading order)", htxt):
            continue
        name = normalize_series_name(htxt)
        lst = h.find_next(lambda tag: tag.name in ("ol", "ul"))
        if not lst:
            continue
        raw_books = [book_from_text(li.get_text(" ", strip=True)) for li in lst.find_all("li", recursive=False)]
        books = [b for b in raw_books if b]
        if books:
            results.append((name, books))
    return results

def main():
    all_series = []
    
    print("=== Order of Books Scraper ===")
    print("Getting author links...")
    
    # Try multiple methods to get author links
    author_links = []
    
    # Method 1: Use sample authors for testing
    print("Using sample author links for testing...")
    author_links = get_sample_author_links()
    
    # Uncomment below to try other methods:
    # Method 2: Try sitemap
    # sitemap_links = get_author_links_from_sitemap()
    # if sitemap_links:
    #     author_links.extend(sitemap_links)
    
    # Method 3: Try pagination
    # if not author_links:
    #     author_links = get_author_links_from_pagination()
    
    if not author_links:
        print("No author links found. Exiting.")
        return
    
    print(f"Found {len(author_links)} author links to process")
    
    # Process authors
    for i, author_url in enumerate(author_links, 1):
        print(f"\n--- Processing author {i}/{len(author_links)} ---")
        try:
            author_series = scrape_author_series(author_url)
            all_series.extend(author_series)
            time.sleep(2)  # Be respectful to the server
        except Exception as e:
            print(f"Error processing {author_url}: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total series collected: {len(all_series)}")
    
    # Save results
    output_file = 'orderofbooks_series.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_series, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {output_file}")
    
    # Show sample of what was collected
    if all_series:
        print(f"\nSample of collected data:")
        for i, series in enumerate(all_series[:3]):
            print(f"{i+1}. {series['series_name']} by {series['author']} ({len(series['books'])} books)")
            for j, book in enumerate(series['books'][:3]):
                print(f"   {j+1}. {book['title']}")
            if len(series['books']) > 3:
                print(f"   ... and {len(series['books']) - 3} more books")

if __name__ == "__main__":
    main()
