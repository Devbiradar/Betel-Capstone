import asyncio
import os
from playwright.async_api import async_playwright

async def take_snapshots():
    # Ensure Playwright chromium is installed if not already
    
    html_files = [
        "results/dashboards/QAH_report.html",
        "results/dashboards/QTL_report.html",
        "results/dashboards/QSVM_report.html"
    ]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1200, 'height': 800})
        
        # set dark background so the transparent edges look correct
        
        for html_file in html_files:
            if not os.path.exists(html_file):
                continue
                
            abs_path = "file:///" + os.path.abspath(html_file).replace('\\', '/')
            await page.goto(abs_path)
            
            # Wait for any potential DOM loading
            await page.wait_for_timeout(500)
            
            # The HTML has a specific body size we want to capture
            # Let's capture the specific container
            await page.evaluate("document.body.style.background = '#0b1120'")
            
            model_id = os.path.basename(html_file).replace("_report.html", "")
            out_png = f"results/dashboards/{model_id}_dashboard.png"
            
            await page.locator('.max-w-6xl').screenshot(path=out_png)
            print(f"Captured {out_png}")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(take_snapshots())
