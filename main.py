from dotenv import load_dotenv
load_dotenv()

from bmo.bmo import BMO


def main():
    BMO().run()


if __name__ == "__main__":
    main()
