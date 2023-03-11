# Using Replit's Secrets Tool
Replit's Secrets Tool allows you to store sensitive information, such as passwords and API keys, in a secure manner. Storing sensitive information, such as API keys, as environment variables provides security, flexibility, and reusability.
Let's checkout the [documentation](https://docs.replit.com/programming-ide/workspace-features/storing-sensitive-information-environment-variables).

## Storing the Huggingface API Key as an Environment Variable
To store the Huggingface API key as an environment variable in Replit:

Go to the "Tools" section in the bottom left corner of the Replit workspace and click on the "Secrets" button (lock icon).

In the "key" box, name the variable "inference_api" and paste the API key in the "value" box.

Click "Add new Secret." Retrieve your [Huggingface API key](https://huggingface.co/settings/tokens) and put it in there. It's now stored securely as an environment variable!

You can access the stored environment variable in your code using the following code:

```python3
import os
API_Token = os.environ['HFKEY']
```