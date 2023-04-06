import urllib.request
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.is_content = False

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.is_content = True

    def handle_endtag(self, tag):
        if tag == "p":
            self.is_content = False

    def handle_data(self, data):
        if self.is_content:
            self.text.append(data.strip())


def main():
    url = "https://en.wikipedia.org/wiki/Stable_Diffusion"
    response = urllib.request.urlopen(url)
    html_content = response.read().decode("utf-8")

    parser = MyHTMLParser()
    parser.feed(html_content)

    extracted_text = " ".join(parser.text)

    with open("output.txt", "w") as output_file:
        output_file.write(extracted_text)

    print("Text extracted and saved to output.txt")


if __name__ == "__main__":
    main()
