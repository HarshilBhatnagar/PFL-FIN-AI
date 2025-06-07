import os
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime
from typing import List, Dict
import json
from pathlib import Path
import re
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFDownloader:
    def __init__(self):
        self.base_dir = Path("data/pdfs")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # List of NBFCs to track with their base URLs
        self.nbfc_configs = {
            "Bajaj Finance": {
                "base_url": "https://www.bajajfinserv.in/investor-relations/financial-results",
                "type": "html"
            },
            "HDFC Bank": {
                "base_url": "https://www.hdfcbank.com/personal/resources/investor-relations/financial-results",
                "type": "html"
            },
            "ICICI Bank": {
                "base_url": "https://www.icicibank.com/aboutus/annual.page",
                "type": "html"
            },
            "Axis Bank": {
                "base_url": "https://www.axisbank.com/investor-relations/financial-results",
                "type": "html"
            },
            "Kotak Mahindra Bank": {
                "base_url": "https://www.kotak.com/en/investor-relations/financial-results.html",
                "type": "html"
            },
            "IDFC First Bank": {
                "base_url": "https://www.idfcfirstbank.com/investor-relations/financial-results",
                "type": "html"
            },
            "Shriram Finance": {
                "base_url": "https://www.shriramfinance.in/investor-relations/financial-results",
                "type": "html"
            },
            "Muthoot Finance": {
                "base_url": "https://www.muthootfinance.com/investor-relations/financial-results",
                "type": "html"
            },
            "Cholamandalam Finance": {
                "base_url": "https://www.cholamandalam.com/investor-relations/financial-results",
                "type": "html"
            },
            "Mahindra Finance": {
                "base_url": "https://www.mahindrafinance.com/investor-relations/financial-results",
                "type": "html"
            }
        }
        
        # Create metadata file
        self.metadata_file = self.base_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"downloads": []}

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_nbfc_urls(self, nbfc: str) -> List[str]:
        """Get URLs for NBFC's financial reports."""
        urls = []
        try:
            config = self.nbfc_configs.get(nbfc)
            if not config:
                logger.error(f"No configuration found for {nbfc}")
                return urls

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(config["base_url"], headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # NBFC-specific scraping logic
            if nbfc == "Bajaj Finance":
                # Look for PDF links in financial results section
                pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
                for link in pdf_links:
                    if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                        urls.append(urljoin(config["base_url"], link['href']))
                        
            elif nbfc == "HDFC Bank":
                # HDFC Bank specific logic
                pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
                for link in pdf_links:
                    if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                        urls.append(urljoin(config["base_url"], link['href']))
                        
            elif nbfc == "ICICI Bank":
                # ICICI Bank specific logic
                pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
                for link in pdf_links:
                    if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                        urls.append(urljoin(config["base_url"], link['href']))
                        
            elif nbfc == "Axis Bank":
                # Axis Bank specific logic
                financial_section = soup.find('div', {'class': 'financial-results'})
                if financial_section:
                    pdf_links = financial_section.find_all('a', href=re.compile(r'\.pdf$', re.I))
                    for link in pdf_links:
                        if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                            urls.append(urljoin(config["base_url"], link['href']))
                            
            elif nbfc == "Kotak Mahindra Bank":
                # Kotak Mahindra Bank specific logic
                results_section = soup.find('div', {'class': 'financial-results-section'})
                if results_section:
                    pdf_links = results_section.find_all('a', href=re.compile(r'\.pdf$', re.I))
                    for link in pdf_links:
                        if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                            urls.append(urljoin(config["base_url"], link['href']))
                            
            elif nbfc == "IDFC First Bank":
                # IDFC First Bank specific logic
                investor_section = soup.find('div', {'class': 'investor-relations'})
                if investor_section:
                    pdf_links = investor_section.find_all('a', href=re.compile(r'\.pdf$', re.I))
                    for link in pdf_links:
                        if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                            urls.append(urljoin(config["base_url"], link['href']))
                            
            elif nbfc == "Shriram Finance":
                # Shriram Finance specific logic
                financial_section = soup.find('div', {'class': 'financial-results'})
                if financial_section:
                    pdf_links = financial_section.find_all('a', href=re.compile(r'\.pdf$', re.I))
                    for link in pdf_links:
                        if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                            urls.append(urljoin(config["base_url"], link['href']))
                            
            elif nbfc == "Muthoot Finance":
                # Muthoot Finance specific logic
                investor_section = soup.find('div', {'class': 'investor-relations'})
                if investor_section:
                    pdf_links = investor_section.find_all('a', href=re.compile(r'\.pdf$', re.I))
                    for link in pdf_links:
                        if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                            urls.append(urljoin(config["base_url"], link['href']))
                            
            elif nbfc == "Cholamandalam Finance":
                # Cholamandalam Finance specific logic
                financial_section = soup.find('div', {'class': 'financial-results'})
                if financial_section:
                    pdf_links = financial_section.find_all('a', href=re.compile(r'\.pdf$', re.I))
                    for link in pdf_links:
                        if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                            urls.append(urljoin(config["base_url"], link['href']))
                            
            elif nbfc == "Mahindra Finance":
                # Mahindra Finance specific logic
                investor_section = soup.find('div', {'class': 'investor-relations'})
                if investor_section:
                    pdf_links = investor_section.find_all('a', href=re.compile(r'\.pdf$', re.I))
                    for link in pdf_links:
                        if any(keyword in link.text.lower() for keyword in ['annual', 'quarterly', 'financial']):
                            urls.append(urljoin(config["base_url"], link['href']))
            
            # Filter out already downloaded URLs
            downloaded_urls = {item['url'] for item in self.metadata['downloads']}
            urls = [url for url in urls if url not in downloaded_urls]
            
        except Exception as e:
            logger.error(f"Error getting URLs for {nbfc}: {str(e)}")
        
        return urls

    def _download_pdf(self, url: str, nbfc: str) -> bool:
        """Download a PDF file from URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, stream=True, headers=headers)
            if response.status_code == 200:
                # Create filename from URL and current timestamp
                filename = f"{nbfc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                filepath = self.base_dir / filename
                
                # Save the PDF
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Update metadata
                self.metadata["downloads"].append({
                    "nbfc": nbfc,
                    "url": url,
                    "filename": filename,
                    "download_date": datetime.now().isoformat()
                })
                self._save_metadata()
                
                logger.info(f"Successfully downloaded {filename}")
                return True
            else:
                logger.error(f"Failed to download from {url}: Status code {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading from {url}: {str(e)}")
            return False

    def download_all_reports(self):
        """Download financial reports for all NBFCs."""
        for nbfc in self.nbfc_configs.keys():
            logger.info(f"Processing {nbfc}")
            urls = self._get_nbfc_urls(nbfc)
            
            for url in urls:
                self._download_pdf(url, nbfc)

def main():
    downloader = PDFDownloader()
    downloader.download_all_reports()

if __name__ == "__main__":
    main() 