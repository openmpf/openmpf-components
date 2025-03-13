import argparse
from google import genai
from PIL import Image
import sys

def main():
    parser = argparse.ArgumentParser(description='Sends image and prompt to Gemini Client for processing.')

    parser.add_argument("--model", "-m", type=str, default="gemini-1.5-pro", help="The name of the Gemini model to use.")
    parser.add_argument("--filepath", "-f", type=str, required=True, help="Path to the media file to process with Gemini.")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="The prompt you want to use with the image.")
    parser.add_argument("--api_key", "-a", type=str, required=True, help="Your API key for Gemini.")
    args = parser.parse_args()
    
    try:
        client = genai.Client(api_key=args.api_key)
        content = client.models.generate_content(model=args.model, contents=[args.prompt, Image.open(args.filepath)])
        print(content.text)
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()