# Compound query hand-check sheet (Part C1)

The assignment asks for 10 rewrites to be hand-checked and fixed where they read badly.
These 10 were picked with `random.Random(7).sample(range(100), 10)`.
For each one: tick **OK**, or write the fix in **Fixed text**. Then copy any fix into
`data/compound_rewrites.json` and re-run `python data/build_compound.py`.

Checklist per item: (1) reads like one natural question, (2) both halves' intents survive,
(3) no new facts added, (4) key terms kept (retrieval fairness).

## c006 (pair cosine -0.012)

- Half A (qid 8289): Is there a website with options chain charts?
- Half B (qid 1003): The difference between Islamic Banks and Western Banks
- **Rewrite:** Is there a website with options chain charts, and what is the difference between Islamic banks and Western banks?
- [ ] OK   - Fixed text: 
- Reviewer notes: 

## c009 (pair cosine 0.108)

- Half A (qid 7396): Tax ID for an international student investing in U.S stocks
- Half B (qid 451): What is bespoke insurance?
- **Rewrite:** What tax ID does an international student need to invest in US stocks, and what is bespoke insurance?
- [ ] OK   - Fixed text: 
- Reviewer notes: 

## c012 (pair cosine 0.167)

- Half A (qid 4425): How does owning a home and paying on a mortgage fit into family savings and investment?
- Half B (qid 806): Trying to understand Return on Capital (Joel Greenblatt's Magic Formula version)
- **Rewrite:** How does owning a home and paying a mortgage fit into family savings and investment, and how is Return on Capital computed in Joel Greenblatt's Magic Formula?
- [ ] OK   - Fixed text: 
- Reviewer notes: 

## c019 (pair cosine 0.244)

- Half A (qid 9005): What is an ideal number of stock positions that I should have in my portfolio?
- Half B (qid 1548): How profitable is selling your customer base?
- **Rewrite:** What is the ideal number of stock positions to hold in my portfolio, and how profitable is it to sell your customer base?
- [ ] OK   - Fixed text: 
- Reviewer notes: 

## c041 (pair cosine 0.126)

- Half A (qid 5096): What effect will the financial reform bill have on everyday Americans?
- Half B (qid 7769): Can capital gains be used to fund an IRA with tax advantages?
- **Rewrite:** What effect will the financial reform bill have on everyday Americans, and can capital gains be used to fund an IRA with tax advantages?
- [ ] OK   - Fixed text: 
- Reviewer notes: 

## c046 (pair cosine 0.199)

- Half A (qid 6760): What are the tax implications of earnings from the stock market (BSE)?
- Half B (qid 6872): Online brokers with a minimum stock purchase lower than $500
- **Rewrite:** What are the tax implications of earnings from the Indian stock market (BSE), and which online brokers have a minimum stock purchase below $500?
- [ ] OK   - Fixed text: 
- Reviewer notes: 

## c050 (pair cosine 0.171)

- Half A (qid 10054): Is there an academic framework for deciding when to sell in-the-money call options?
- Half B (qid 3300): At what point should I begin paying off student loans?
- **Rewrite:** Is there an academic framework for deciding when to sell in-the-money call options, and at what point should I start paying off my student loans?
- [ ] OK   - Fixed text: 
- Reviewer notes: 

## c068 (pair cosine -0.028)

- Half A (qid 5508): what is shareholders' Equity in balance sheets?
- Half B (qid 4111): Can I predict if it is better to save money in USD or local currency?
- **Rewrite:** What is shareholders' equity on a balance sheet, and can I predict whether it's better to save in USD or in my local currency?
- [ ] OK   - Fixed text: 
- Reviewer notes: 

## c074 (pair cosine -0.01)

- Half A (qid 1938): How to distinguish gift from payment for the service?
- Half B (qid 5381): Question about large capital gain
- **Rewrite:** How do I distinguish a gift from a payment for a service, and what should I know about a large capital gain?
- [ ] OK   - Fixed text: 
- Reviewer notes: 

## c083 (pair cosine 0.07)

- Half A (qid 1658): Do any publically available documents from IR or SEC include all patents the company holds?
- Half B (qid 6913): Why futures has a mark to market concept that is not present in stocks
- **Rewrite:** Do public IR or SEC documents list all the patents a company holds, and why do futures have mark-to-market when stocks don't?
- [ ] OK   - Fixed text: 
- Reviewer notes: 
