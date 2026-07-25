import os
from pathlib import Path
from dotenv import load_dotenv
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def _build_api_instance():
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key["api-key"] = os.getenv("BREVO_API_KEY")
    return sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))


def send_welcome_email(email: str, username: str):
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

    api_key = os.getenv("BREVO_API_KEY")
    sender_name = os.getenv("SENDER_NAME")
    sender_email = os.getenv("SENDER_EMAIL")

    if not api_key or not sender_name or not sender_email:
        print("Warning: welcome email not sent because Brevo configuration is incomplete.")
        return

    try:
        api_instance = _build_api_instance()
        email_data = sib_api_v3_sdk.SendSmtpEmail(
            sender={
                "name": sender_name,
                "email": sender_email,
            },
            to=[
                {
                    "email": email,
                    "name": username,
                }
            ],
            subject="Welcome to Laptop RAG 🎉",
            html_content=f"""
            <html>
                <body>
                    <h2>Welcome, {username}! 👋</h2>

                    <p>
                        Thank you for signing up for <b>Laptop RAG</b>.
                    </p>

                    <p>
                        Your account has been created successfully.
                    </p>

                    <p>
                        We hope you enjoy using our platform.
                    </p>

                    <br>

                    <p>
                        Regards,<br>
                        Laptop RAG Team
                    </p>
                </body>
            </html>
            """
        )

        response = api_instance.send_transac_email(email_data)
        print("Welcome email sent.")
        print(response)

    except ApiException as e:
        print("Brevo Error:", e)
    except Exception as e:
        print("Unexpected email error:", e)
