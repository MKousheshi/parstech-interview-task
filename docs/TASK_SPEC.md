# Task: Business Consultant Chatbot Flow (LangChain / LangGraph)

*This file is a faithful English rendering of the original task description, kept as close to the source wording as possible for reference. Open questions and scope clarifications live separately in `CLARIFICATIONS.md`; architecture and scope decisions live in `DECISIONS.md`.*

## Title

Set up a chatbot flow for a customer on the topic of "business consultant", using the LangChain and LangGraph frameworks.

## Context and assumptions

Assume there is a store selling products related to digital marketing. The store's product list is provided as an attached JSON file.

## Core features

The chatbot must implement the following two main capabilities:

### 1. Product search

Ability to search over the store's product list.

### 2. Business consultation

The consultation process must follow this flow:

1. **Information gathering** — During the conversation, the following 4 main entities must be collected from the user, in this order:
   1. Type of business
   2. Type of customers (B2B or B2C)
   3. Geographic location
   4. Whether they have a virtual sales channel (website or page)
2. **Initial analysis** — Once all 4 items have been collected, use the model's general/free knowledge (no dedicated knowledge base required) to provide an initial analysis and recommendation to the user.
3. **Product suggestion** — Following the analysis, suggest relevant products and sales-boosting packages from the store to the user.

## Tech stack

- LangChain
- LangGraph
