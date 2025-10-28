from playwright.sync_api import sync_playwright
from typer import Typer

cli = Typer()


@cli.command()
def main(url: str):
    custom_referer = "https://s.epinionglobal.com/mrIWeb/mrIWeb.srf?I.Project=P2100211_TEST&i.user2=n&i.user1=XXXXX"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set to True to run headless
        context = browser.new_context()

        page = context.new_page()

        page.goto(url, referer=custom_referer)
        input("Press any key to continue...")


if __name__ == "__main__":
    cli()
