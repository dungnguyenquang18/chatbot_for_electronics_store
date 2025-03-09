# Chatbot for Electronics Store

This project is a chatbot designed to assist customers in an electronics store. The chatbot can answer questions, provide product recommendations, and help with various customer service tasks.

## Project Structure

## Getting Started

### Prerequisites

- Python >= 3.9
- Required Python packages (listed in `requirements.txt`)
- MongoDB

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/chatbot_for_electronics_store.git
   ```
2. Navigate to the project directory:
   ```sh
   cd chatbot_for_electronics_store
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Chatbot

1. First, run the cells in the `crawl_data.ipynb` file located in the `build_data` folder to crawl data and insert it into MongoDB:

   ```sh
   jupyter notebook build_data/crawl_data.ipynb
   ```

   Open the notebook in your browser and run all the cells.

2. After the data has been inserted into MongoDB, start the chatbot server by running the following command:
   ```sh
   python chatbot/serve.py
   ```

The chatbot server should now be running and ready to handle requests.
