"""Question templates for every tool.

{field} is replaced with a rendered value; {field:n} forces a bare number;
{field:adv|adj|every} picks a frequency wording; {field:money} renders as dollars.

train    -- used to generate training data
test     -- held out: never seen in training, so the test set measures
            generalization to new phrasings rather than memorization
partial  -- (template, missing fields): questions that leave something out,
            used to teach the model to ask instead of guessing
"""

TEMPLATES = {
    "compound_interest": {
        "train": [
            "If I invest {principal} at {annual_rate} for {years}, how much will I have?",
            "How much will {principal} grow to in {years} at {annual_rate} interest compounded {compounds_per_year}?",
            "I have {principal} in a CD paying {annual_rate} a year. What will it be worth after {years}?",
            "What's the future value of {principal} invested at {annual_rate} for {years}?",
            "My grandparents gave me {principal}. If I leave it in an account earning {annual_rate} compounded {compounds_per_year}, what will I have in {years}?",
            "{principal} at {annual_rate}, compounded {compounds_per_year}, for {years}. Final balance?",
            "Calculate compound interest on {principal} at {annual_rate} per year over {years}.",
            "How much interest will {principal} earn in {years} if it compounds {compounds_per_year} at {annual_rate}?",
            "I'm putting {principal} into a high-yield savings account at {annual_rate} APY and not touching it for {years}. How much will be there?",
            "what does {principal} turn into after {years} at {annual_rate}",
            "Lump sum of {principal}, {annual_rate} annual return, {years} horizon. What's it worth at the end?",
        ],
        "test": [
            "Suppose I deposit {principal} today at an interest rate of {annual_rate}, compounded {compounds_per_year}. How much will the account hold in {years}?",
            "If {principal} sits in an index fund averaging {annual_rate} a year for {years}, what's the ending value?",
            "Help me figure out the growth of a one-time {principal} investment over {years} at {annual_rate}.",
        ],
        "partial": [
            ("If I invest {principal} at {annual_rate}, how much will I have?", ["years"]),
            ("How much will {principal} grow to in {years}?", ["annual_rate"]),
            ("What will my savings be worth in {years} at {annual_rate} if I invest a lump sum now?", ["principal"]),
        ],
    },
    "simple_interest": {
        "train": [
            "What's the simple interest on {principal} at {annual_rate} for {years}?",
            "I lent my brother {principal} at {annual_rate} simple interest for {years}. How much interest does he owe me?",
            "Calculate simple interest: principal {principal}, rate {annual_rate}, time {years}.",
            "How much will I earn on {principal} over {years} at {annual_rate} if the interest is simple, not compounded?",
            "A bond pays {annual_rate} simple interest on {principal}. What's the total interest after {years}?",
            "If a loan of {principal} charges {annual_rate} simple interest per year, what will I repay in total after {years}?",
            "simple interest {principal} {annual_rate} {years}",
            "My bank quotes {annual_rate} simple annual interest on a {principal} deposit. How much interest after {years}?",
            "Using simple interest, what does {principal} earn in {years} at {annual_rate} a year?",
        ],
        "test": [
            "How much simple interest accrues on {principal} borrowed at {annual_rate} per year for {years}?",
            "I'm owed {principal} plus {annual_rate} simple interest per year. What's the payoff after {years}?",
            "Find the non-compounding interest on {principal} at {annual_rate} over {years}.",
        ],
        "partial": [
            ("What's the simple interest on {principal} at {annual_rate}?", ["years"]),
            ("How much simple interest would I earn over {years} at {annual_rate}?", ["principal"]),
        ],
    },
    "future_value_with_contributions": {
        "train": [
            "If I invest {contribution} every month at {annual_rate} for {years}, how much will I have?",
            "I'm contributing {contribution} {contributions_per_year:every} to my 401(k). Assuming {annual_rate} returns, what will it be worth in {years}?",
            "I already have {initial_balance} saved and plan to add {contribution} a month. With a {annual_rate} annual return, how much will I have after {years}?",
            "How much will I accumulate by saving {contribution} per month for {years} at {annual_rate}?",
            "Starting with {initial_balance}, adding {contribution} {contributions_per_year:every}, earning {annual_rate} a year. Balance after {years}?",
            "If I put {contribution} {contributions_per_year:every} into a Roth IRA earning {annual_rate}, what will it grow to in {years}?",
            "My plan: invest {contribution} monthly for {years}. Expected return {annual_rate}. Ending balance?",
            "What's the future value of {contribution} monthly deposits at {annual_rate} over {years}?",
            "I'm 25 and want to invest {contribution} a month until I retire in {years}. At {annual_rate} average growth, how much will I have?",
            "How big will my brokerage account be in {years} if it has {initial_balance} now, I add {contribution} {contributions_per_year:every}, and it earns {annual_rate}?",
            "save {contribution}/month at {annual_rate} for {years}, total?",
        ],
        "test": [
            "Suppose I set aside {contribution} {contributions_per_year:every} in an account yielding {annual_rate}. What will I have after {years}?",
            "With {initial_balance} already invested plus monthly contributions of {contribution}, what's my projected balance in {years} at {annual_rate}?",
            "Projected value of auto-investing {contribution} each month for {years}, assuming {annual_rate} growth?",
        ],
        "partial": [
            ("If I invest {contribution} every month for {years}, how much will I have?", ["annual_rate"]),
            ("If I save {contribution} a month at {annual_rate}, what will I end up with?", ["years"]),
        ],
    },
    "present_value": {
        "train": [
            "How much do I need to invest today to have {future_value} in {years} at {annual_rate}?",
            "What's the present value of {future_value} received {years} from now if the discount rate is {annual_rate}?",
            "I want {future_value} for my kid's college in {years}. If I can earn {annual_rate} compounded {compounds_per_year}, how much should I put away now as a lump sum?",
            "What lump sum today grows to {future_value} in {years} at {annual_rate} a year?",
            "Discount {future_value} back {years} at {annual_rate}. What's it worth today?",
            "Someone promises me {future_value} in {years}. With a {annual_rate} rate, what is that worth in today's money?",
            "PV of {future_value}, {years} out, {annual_rate} rate?",
            "How much should I deposit now at {annual_rate} interest, compounded {compounds_per_year}, to reach {future_value} after {years}?",
            "To end up with {future_value} in {years}, what's the one-time amount I need to invest now if it earns {annual_rate}?",
        ],
        "test": [
            "A zero-coupon bond pays {future_value} at maturity in {years}. At a {annual_rate} yield, what's it worth today?",
            "What single deposit would become {future_value} in {years} at {annual_rate} compounded {compounds_per_year}?",
            "Present value calculation: {future_value} due in {years}, interest rate {annual_rate}.",
        ],
        "partial": [
            ("How much do I need to invest today to have {future_value} at {annual_rate}?", ["years"]),
            ("What lump sum today grows to {future_value} in {years}?", ["annual_rate"]),
        ],
    },
    "present_value_of_annuity": {
        "train": [
            "What's the present value of receiving {payment} a month for {years} at a {annual_rate} discount rate?",
            "I'm offered a pension of {payment} per month for {years}. At {annual_rate}, what is that worth as a lump sum today?",
            "How much is an annuity paying {payment} {payments_per_year:every} for {years} worth today if rates are {annual_rate}?",
            "Lottery question: I can take {payment} {payments_per_year:every} for {years}. Using a {annual_rate} discount rate, what's the equivalent lump sum?",
            "Present value of {payment} monthly payments over {years}, discounted at {annual_rate}?",
            "A settlement pays {payment} {payments_per_year:every} for {years}. With money earning {annual_rate}, what is the stream worth now?",
            "What lump sum today equals getting {payment} per month for {years}, given {annual_rate} interest?",
            "How much would I need today to fund withdrawals of {payment} a month for {years} if the money earns {annual_rate}?",
        ],
        "test": [
            "My structured settlement pays {payment} {payments_per_year:every} for the next {years}. Discounting at {annual_rate}, what's its value today?",
            "Value today of collecting {payment} a month for {years} at {annual_rate}?",
            "If a rental property nets {payment} monthly for {years}, what's that income stream worth now at a {annual_rate} rate?",
        ],
        "partial": [
            ("What's the present value of {payment} a month for {years}?", ["annual_rate"]),
            ("I'm offered {payment} a month as a pension. What's it worth today at {annual_rate}?", ["years"]),
        ],
    },
    "continuous_compounding": {
        "train": [
            "How much will {principal} grow to at {annual_rate} compounded continuously for {years}?",
            "Continuous compounding: {principal} at {annual_rate} for {years}. Future value?",
            "If interest compounds continuously at {annual_rate}, what does {principal} become after {years}?",
            "What's the value of {principal} after {years} with continuously compounded interest of {annual_rate}?",
            "Using e^(rt), what is {principal} worth in {years} at {annual_rate}?",
            "My textbook problem: {principal} invested at {annual_rate} compounded continuously. Find the amount after {years}.",
            "{principal}, continuous compounding, {annual_rate}, {years}",
            "How much interest does {principal} earn over {years} at {annual_rate} if compounding is continuous?",
        ],
        "test": [
            "Find the future value of {principal} under continuous compounding at {annual_rate} after {years}.",
            "With interest compounded continuously at {annual_rate}, how big is a {principal} deposit after {years}?",
            "Pert formula: P = {principal:n}, r = {annual_rate}, t = {years:n} years. What's A?",
        ],
        "partial": [
            ("How much will {principal} grow to compounded continuously for {years}?", ["annual_rate"]),
        ],
    },
    "effective_annual_rate": {
        "train": [
            "What's the effective annual rate of {nominal_rate} compounded {compounds_per_year}?",
            "Convert an APR of {nominal_rate} with {compounds_per_year:adj} compounding to APY.",
            "My credit card charges {nominal_rate} APR compounded {compounds_per_year}. What's the actual yearly rate?",
            "If a savings account pays {nominal_rate} compounded {compounds_per_year}, what's the APY?",
            "Effective interest rate for a nominal {nominal_rate} rate, compounding {compounds_per_year}?",
            "{nominal_rate} nominal, compounded {compounds_per_year}. EAR?",
            "How much is {nominal_rate} compounded {compounds_per_year} really, on an annual basis?",
            "What's the true annual cost of a loan at {nominal_rate} compounded {compounds_per_year}?",
        ],
        "test": [
            "Turn a stated rate of {nominal_rate}, compounded {compounds_per_year}, into an effective annual yield.",
            "A bank advertises {nominal_rate} compounded {compounds_per_year}. What's the equivalent annual percentage yield?",
            "APR {nominal_rate}, compounding {compounds_per_year}: what's the effective rate?",
        ],
        "partial": [
            ("What's the effective annual rate of {nominal_rate}?", ["compounds_per_year"]),
            ("What's the APY on a savings account that compounds {compounds_per_year}?", ["nominal_rate"]),
        ],
    },
    "rule_of_72": {
        "train": [
            "How long will it take to double my money at {annual_rate}?",
            "Using the rule of 72, how many years to double at {annual_rate} interest?",
            "At {annual_rate} a year, when does an investment double?",
            "Rule of 72 for {annual_rate}?",
            "If my portfolio returns {annual_rate} annually, how many years until it doubles?",
            "How quickly does money double at a {annual_rate} return?",
            "What's the doubling time for an investment earning {annual_rate}?",
            "I heard about the rule of 72. What does it say for a {annual_rate} rate?",
            "years to double money {annual_rate}",
        ],
        "test": [
            "Roughly how many years does it take for savings to double if they earn {annual_rate}?",
            "At an interest rate of {annual_rate}, estimate the time to double using rule of 72.",
            "When will my investment be twice as big if it grows {annual_rate} per year?",
        ],
        "partial": [
            ("How long will it take to double my money?", ["annual_rate"]),
        ],
    },
    "years_to_goal": {
        "train": [
            "How long will it take for {current_amount} to grow to {target_amount} at {annual_rate}?",
            "I have {current_amount} invested earning {annual_rate}. How many years until it reaches {target_amount}?",
            "When will my {current_amount} turn into {target_amount} if it grows {annual_rate} a year?",
            "How many years to go from {current_amount} to {target_amount} at {annual_rate} compounded {compounds_per_year}?",
            "My savings are at {current_amount}. If they earn {annual_rate} and I don't add anything, when will I hit {target_amount}?",
            "Time needed for {current_amount} to become {target_amount} with {annual_rate} annual growth?",
            "{current_amount} -> {target_amount} at {annual_rate}: how many years?",
            "I want my {current_amount} portfolio to reach {target_amount}. At {annual_rate} returns, how long will that take?",
        ],
        "test": [
            "Assuming {annual_rate} growth compounded {compounds_per_year}, how long until {current_amount} becomes {target_amount}?",
            "How many years before my {current_amount} nest egg grows into {target_amount} at {annual_rate}?",
            "Estimate the time for an account to grow from {current_amount} to {target_amount} earning {annual_rate} per year.",
        ],
        "partial": [
            ("How long will it take for {current_amount} to grow to {target_amount}?", ["annual_rate"]),
            ("How long until my savings reach {target_amount} at {annual_rate}?", ["current_amount"]),
        ],
    },
    "inflation_adjusted_value": {
        "train": [
            "With {inflation_rate} inflation, what will {amount} be worth in {years}?",
            "How much will something that costs {amount} today cost in {years} if inflation is {inflation_rate}?",
            "What's the purchasing power of {amount} after {years} of {inflation_rate} inflation?",
            "If inflation averages {inflation_rate}, what will {amount} buy in {years}?",
            "Adjust {amount} for {inflation_rate} annual inflation over {years}.",
            "My salary is {amount}. If inflation runs at {inflation_rate} for {years}, what will I need to earn to keep up?",
            "Inflation {inflation_rate}, {years}, {amount}: future cost and real value?",
            "How much will a {amount} car cost in {years} with {inflation_rate} inflation per year?",
            "In today's dollars, what is {amount} received {years} from now worth if inflation is {inflation_rate}?",
        ],
        "test": [
            "Assuming prices rise {inflation_rate} a year, what will today's {amount} expenses cost in {years}?",
            "How does {inflation_rate} yearly inflation erode {amount} over {years}?",
            "What will {amount} of groceries cost after {years} at an inflation rate of {inflation_rate}?",
        ],
        "partial": [
            ("How much will something that costs {amount} today cost in {years}?", ["inflation_rate"]),
        ],
    },
    "loan_payment": {
        "train": [
            "What's the monthly payment on a {principal} loan at {annual_rate} for {years}?",
            "I'm buying a car for {principal} with a {years:n}-year loan at {annual_rate}. What will my payments be?",
            "Calculate the payment for borrowing {principal} over {years} at {annual_rate} interest.",
            "How much will I pay each month on a {principal} personal loan with a {annual_rate} APR and a {years:n}-year term?",
            "If I take out a {principal} student loan at {annual_rate} and repay it over {years}, what's the monthly payment and total interest?",
            "Loan: {principal}, rate {annual_rate}, term {years}. Monthly payment?",
            "How much interest will I pay in total on a {principal} loan at {annual_rate} for {years}?",
            "What would my {payments_per_year:adj} payment be on {principal} borrowed at {annual_rate} for {years}?",
            "My bank approved me for {principal} at {annual_rate} over {years}. What does that cost per month?",
            "What's the monthly payment on a {principal} car loan for {term_months} at {annual_rate}?",
            "{term_months:n}-month auto loan, {principal} financed at {annual_rate} APR. Payment?",
            "I need to borrow {principal} for a kitchen remodel. At {annual_rate} for {years}, what would I pay monthly?",
        ],
        "test": [
            "What are the monthly installments on {principal} financed at {annual_rate} for {years}?",
            "A lender offers {principal} at {annual_rate} APR, repayable over {term_months}. How much is each payment and what's the total cost?",
            "Figure out the payment on a {years:n}-year, {principal} loan with an interest rate of {annual_rate}.",
        ],
        "partial": [
            ("What's the monthly payment on a {principal} loan at {annual_rate}?", ["years"]),
            ("What's the monthly payment on a {principal} loan over {years}?", ["annual_rate"]),
            ("What would my payments be on a car loan at {annual_rate} for {years}?", ["principal"]),
        ],
    },
    "mortgage_payment": {
        "train": [
            "What would my monthly mortgage payment be on a {home_price} house with {down_payment} down at {annual_rate} for {years}?",
            "I'm looking at a {home_price} home. If I put {down_payment_percent} down and get a {years:n}-year mortgage at {annual_rate}, what's the payment?",
            "Mortgage calc: price {home_price}, {down_payment_percent} down, {annual_rate}, {years:n}-year fixed, property taxes {annual_property_tax} a year, insurance {annual_insurance} a year.",
            "Home price {home_price}, down payment {down_payment}, rate {annual_rate}, {years:n}-year term, HOA {monthly_hoa} a month. What's my total monthly housing payment?",
            "How much would a {home_price} condo cost per month with {down_payment_percent} down, a {annual_rate} rate over {years}, and {monthly_hoa} monthly HOA dues?",
            "We want to buy a house for {home_price}. We have {down_payment} for a down payment and were quoted {annual_rate} on a {years:n}-year loan. Monthly payment?",
            "Estimate PITI on a {home_price} home, {down_payment_percent} down, {annual_rate} {years:n}-year mortgage, {annual_property_tax} annual property tax, {annual_insurance} annual homeowners insurance.",
            "If I buy a {home_price} house with no down payment at {annual_rate} for {years}, what's the monthly payment?",
            "Monthly payment for a {years:n}-year mortgage at {annual_rate} on a {home_price} house after putting {down_payment} down?",
            "{home_price} house, {down_payment_percent} down, {annual_rate} for {years}, taxes {annual_property_tax}/yr. payment?",
        ],
        "test": [
            "We're house hunting around {home_price}. With {down_payment_percent} down and a {annual_rate} {years:n}-year fixed rate, what's our monthly principal and interest?",
            "Total monthly cost of a {home_price} home: {down_payment} down, {annual_rate} over {years}, {annual_property_tax} per year in taxes and {annual_insurance} per year for insurance?",
            "Help me estimate my mortgage: purchase price {home_price}, putting down {down_payment}, rate {annual_rate}, term {years}.",
        ],
        "partial": [
            ("What would my monthly mortgage payment be on a {home_price} house with {down_payment_percent} down?", ["annual_rate", "years"]),
            ("What's the mortgage payment on a {home_price} home at {annual_rate} with {down_payment} down?", ["years"]),
        ],
    },
    "remaining_loan_balance": {
        "train": [
            "I took out a {principal} mortgage at {annual_rate} for {years}. How much will I still owe after {elapsed_years}?",
            "What's the remaining balance on a {principal}, {years:n}-year loan at {annual_rate} after {elapsed_months} of payments?",
            "After {elapsed_years} of payments on a {principal} loan at {annual_rate} over {years}, what's left to pay off?",
            "My car loan was {principal} at {annual_rate} for {years}. I've made {elapsed_months:n} payments. What's the payoff amount?",
            "Remaining principal on {principal} borrowed at {annual_rate}, {years:n}-year term, {elapsed_years} in?",
            "How much principal will I have paid down on my {principal} {annual_rate} {years:n}-year mortgage after {elapsed_years}?",
            "If I sell my house {elapsed_years} into a {years:n}-year, {principal} mortgage at {annual_rate}, how much do I still owe the bank?",
            "Loan balance after {elapsed_months}: original {principal}, {annual_rate}, {years}.",
        ],
        "test": [
            "What will the outstanding balance be {elapsed_years} into a {years:n}-year {principal} loan at {annual_rate}?",
            "I've been paying a {principal} loan ({annual_rate}, {years}) for {elapsed_months}. How much is left?",
            "Payoff balance after {elapsed_months:n} monthly payments on {principal} at {annual_rate} over {years}?",
        ],
        "partial": [
            ("How much will I still owe on my {principal} mortgage after {elapsed_years}?", ["annual_rate", "years"]),
        ],
    },
    "loan_payoff_time": {
        "train": [
            "How long will it take to pay off {balance} on a credit card at {annual_rate} APR if I pay {monthly_payment} a month?",
            "I owe {balance} at {annual_rate}. If I pay {monthly_payment} per month, when will it be paid off?",
            "How many months to pay off a {balance} loan at {annual_rate} with {monthly_payment} monthly payments?",
            "My student loan balance is {balance} at {annual_rate}. Paying {monthly_payment} a month, how long until I'm debt-free?",
            "Credit card: {balance} balance, {annual_rate} APR, paying {monthly_payment}/month. Payoff time and total interest?",
            "If I put {monthly_payment} a month toward my {balance} debt at {annual_rate} interest, how long will it take and how much interest will I pay?",
            "payoff time {balance} {annual_rate} {monthly_payment} per month",
            "I can afford {monthly_payment} a month. How long to clear a {balance} balance charging {annual_rate}?",
            "When will my {balance} car loan at {annual_rate} be paid off if I pay {monthly_payment} every month?",
        ],
        "test": [
            "Paying {monthly_payment} monthly on {balance} of debt at {annual_rate} APR, how many months until it's gone?",
            "How long does it take to eliminate a {balance} balance at {annual_rate} interest with payments of {monthly_payment} per month?",
            "I've got {balance} on my card at {annual_rate}. At {monthly_payment} a month, how many months until I'm debt-free?",
        ],
        "partial": [
            ("How long will it take to pay off {balance} if I pay {monthly_payment} a month?", ["annual_rate"]),
            ("I owe {balance} at {annual_rate}. When will it be paid off?", ["monthly_payment"]),
        ],
    },
    "extra_payment_savings": {
        "train": [
            "If I pay an extra {extra_monthly_payment} a month on my {principal} mortgage at {annual_rate} for {years}, how much interest do I save?",
            "How much faster will I pay off a {principal}, {years:n}-year loan at {annual_rate} if I add {extra_monthly_payment} to each payment?",
            "What happens if I pay {extra_monthly_payment} extra per month on a {principal} loan at {annual_rate} over {years}?",
            "Extra payments: loan {principal}, {annual_rate}, {years:n} years, extra {extra_monthly_payment}/month. Time and interest saved?",
            "I have a {principal} student loan at {annual_rate} with a {years:n}-year term. Is it worth adding {extra_monthly_payment} a month? How much would I save?",
            "How many months would I cut off my {years:n}-year {principal} mortgage at {annual_rate} by paying {extra_monthly_payment} more each month?",
            "Adding {extra_monthly_payment} monthly to a {principal} {annual_rate} loan over {years}: interest savings?",
            "my car loan is {principal} at {annual_rate} for {years}. if i throw an extra {extra_monthly_payment} at it every month how much sooner is it done",
        ],
        "test": [
            "Show me the savings from prepaying {extra_monthly_payment} a month on a {principal} loan at {annual_rate} for {years}.",
            "By how much would an additional {extra_monthly_payment} monthly shorten a {years:n}-year, {annual_rate} mortgage of {principal}?",
            "If I round up my payment by {extra_monthly_payment} on a {principal} loan ({annual_rate}, {years}), what do I save in interest?",
        ],
        "partial": [
            ("If I pay an extra {extra_monthly_payment} a month on my mortgage, how much interest do I save?", ["principal", "annual_rate", "years"]),
        ],
    },
    "max_affordable_loan": {
        "train": [
            "How much can I borrow if I can afford {monthly_payment} a month at {annual_rate} for {years}?",
            "What's the biggest loan I can take with a {monthly_payment} monthly payment, {annual_rate} rate and {years:n}-year term?",
            "I can handle {monthly_payment} per month for a car. At {annual_rate} over {term_months}, how big a car loan can I afford?",
            "Max mortgage for a {monthly_payment} monthly payment at {annual_rate}, {years:n} years?",
            "If my budget for loan payments is {monthly_payment} a month, how much can I finance at {annual_rate} for {years}?",
            "With {monthly_payment} a month going to a {years:n}-year loan at {annual_rate}, what loan amount does that support?",
            "how big a loan does {monthly_payment}/mo get me at {annual_rate} for {years}",
            "We can spend {monthly_payment} monthly on principal and interest. With rates at {annual_rate} on a {years:n}-year mortgage, how much can we borrow?",
            "I can pay {monthly_payment} a month on a {term_months:n}-month car loan at {annual_rate}. How much car can I finance?",
        ],
        "test": [
            "What loan principal corresponds to payments of {monthly_payment} a month at {annual_rate} for {years}?",
            "If I want my payment to stay at {monthly_payment} monthly, how much can I borrow over {term_months} at {annual_rate}?",
            "Affordability check: {monthly_payment}/month, {annual_rate} interest, {years:n}-year term. Maximum loan?",
        ],
        "partial": [
            ("How much can I borrow if I can afford {monthly_payment} a month?", ["annual_rate", "years"]),
        ],
    },
    "refinance_breakeven": {
        "train": [
            "Should I refinance? I owe {current_balance} at {current_rate} with {remaining_years} left. I can get {new_rate} for {new_term_years} with {closing_costs} in closing costs. What's the break-even?",
            "Refinancing {current_balance} from {current_rate} ({remaining_years} remaining) to {new_rate} over {new_term_years}, closing costs {closing_costs}. Monthly savings and break-even?",
            "My mortgage balance is {current_balance} at {current_rate} with {remaining_years} to go. If I refi into a {new_term_years:n}-year loan at {new_rate} and pay {closing_costs} in fees, how long until it pays for itself?",
            "How much would I save per month refinancing {current_balance} from {current_rate} to {new_rate}? Remaining term {remaining_years}, new term {new_term_years}.",
            "Refi analysis: balance {current_balance}, current rate {current_rate}, {remaining_years} left; new rate {new_rate}, new term {new_term_years}; costs {closing_costs}.",
            "If I refinance my {current_balance} loan at {current_rate} to {new_rate} on a new {new_term_years:n}-year term with {closing_costs} closing costs, when do I break even? {remaining_years} remain on the current loan.",
            "Is a refinance from {current_rate} to {new_rate} worth it on {current_balance}? {remaining_years} left on my loan, the new loan would be {new_term_years}, closing costs are {closing_costs}.",
        ],
        "test": [
            "Compare keeping my {current_balance} mortgage at {current_rate} ({remaining_years} left) versus refinancing to {new_rate} for {new_term_years} with {closing_costs} of costs.",
            "What's the break-even point on a refi: {current_balance} owed, {current_rate} now, {new_rate} offered, {remaining_years} remaining, {new_term_years} new term, {closing_costs} in fees?",
            "I could lower my rate from {current_rate} to {new_rate} on {current_balance} ({remaining_years} remaining), new term {new_term_years}, closing costs {closing_costs}. Monthly savings?",
        ],
        "partial": [
            ("Should I refinance my {current_balance} mortgage from {current_rate} to {new_rate}?", ["remaining_years", "new_term_years"]),
        ],
    },
    "roi": {
        "train": [
            "I invested {initial_investment} and now it's worth {final_value}. What's my ROI?",
            "What's the return on investment if I bought for {initial_investment} and sold for {final_value}?",
            "Calculate ROI: cost {initial_investment}, current value {final_value}.",
            "My {initial_investment} investment grew to {final_value}. What percentage return is that?",
            "I put {initial_investment} into crypto and it's now worth {final_value}. How much did I make or lose, in percent?",
            "Bought shares for {initial_investment} total, sold them for {final_value}. Return on investment?",
            "ROI on {initial_investment} -> {final_value}?",
            "We spent {initial_investment} on a marketing campaign that brought in {final_value} in revenue. What's the ROI?",
            "I flipped a car: paid {initial_investment}, sold for {final_value}. What's my return?",
        ],
        "test": [
            "An investment of {initial_investment} is now valued at {final_value}. What's the total return in percent?",
            "Total ROI if my portfolio went from {initial_investment} invested to {final_value}?",
            "I paid {initial_investment} for some collectibles and they appraise at {final_value}. What's my return on investment?",
        ],
        "partial": [
            ("I invested {initial_investment} a few years ago. What's my ROI?", ["final_value"]),
        ],
    },
    "cagr": {
        "train": [
            "My investment grew from {beginning_value} to {ending_value} over {years}. What's the annual growth rate?",
            "What's the CAGR if revenue went from {beginning_value} to {ending_value} in {years}?",
            "Calculate compound annual growth rate: start {beginning_value}, end {ending_value}, {years}.",
            "My portfolio was {beginning_value} {years} ago and it's {ending_value} now. What average annual return is that?",
            "If a stock went from {beginning_value} to {ending_value} over {years}, what's its annualized return?",
            "Annualized growth from {beginning_value} to {ending_value} across {years}?",
            "CAGR {beginning_value} to {ending_value}, {years:n} yrs",
            "My house cost {beginning_value} and sold for {ending_value} after {years}. What was the yearly appreciation rate?",
        ],
        "test": [
            "What yearly growth rate turns {beginning_value} into {ending_value} in {years}?",
            "Our company's sales rose from {beginning_value} to {ending_value} over {years}. What's the compound annual growth rate?",
            "Find the annualized return for an asset that went from {beginning_value} to {ending_value} in {years}.",
        ],
        "partial": [
            ("My investment grew from {beginning_value} to {ending_value}. What's the annual growth rate?", ["years"]),
        ],
    },
    "npv": {
        "train": [
            "What's the NPV of a project that costs {initial_investment} and returns {cash_flows}, at a {discount_rate} discount rate?",
            "Net present value: initial investment {initial_investment}, cash flows {cash_flows}, discount rate {discount_rate}.",
            "Is this investment worth it? It costs {initial_investment} upfront and brings in {cash_flows}. My required return is {discount_rate}.",
            "Calculate NPV at {discount_rate} for an outlay of {initial_investment} with cash inflows of {cash_flows}.",
            "We're considering new equipment for {initial_investment}. It should generate {cash_flows}. Using a {discount_rate} cost of capital, what's the NPV?",
            "NPV? cost {initial_investment}, flows {cash_flows}, rate {discount_rate}",
            "A rental property costs {initial_investment} and should produce {cash_flows}. Discounting at {discount_rate}, what's the net present value?",
            "What's the net present value at {discount_rate} if I invest {initial_investment} and receive {cash_flows}?",
        ],
        "test": [
            "Evaluate a project with an initial cost of {initial_investment} and expected cash flows of {cash_flows} using a {discount_rate} hurdle rate. What's the NPV?",
            "Discounting at {discount_rate}, is a {initial_investment} investment returning {cash_flows} positive NPV?",
            "Find the net present value: spend {initial_investment} today, get back {cash_flows}, discount rate {discount_rate}.",
        ],
        "partial": [
            ("What's the NPV of a project that costs {initial_investment} and returns {cash_flows}?", ["discount_rate"]),
        ],
    },
    "irr": {
        "train": [
            "What's the IRR of an investment that costs {initial_investment} and pays {cash_flows}?",
            "Calculate the internal rate of return: initial outlay {initial_investment}, cash flows {cash_flows}.",
            "I'm buying a business for {initial_investment} that should return {cash_flows}. What's the IRR?",
            "IRR for {initial_investment} invested with returns of {cash_flows}?",
            "What annual return am I really getting if I pay {initial_investment} and get back {cash_flows}?",
            "A project needs {initial_investment} upfront and generates {cash_flows}. What's its internal rate of return?",
            "irr: -{initial_investment:n} now, then {cash_flows}",
            "If I invest {initial_investment} in solar panels that save {cash_flows}, what's the IRR?",
        ],
        "test": [
            "Determine the IRR for a deal requiring {initial_investment} that returns {cash_flows}.",
            "What discount rate makes the NPV zero for an initial cost of {initial_investment} and cash flows of {cash_flows}?",
            "Internal rate of return on {initial_investment} with future inflows of {cash_flows}?",
        ],
        "partial": [
            ("What's the IRR of an investment that costs {initial_investment}?", ["cash_flows"]),
        ],
    },
    "payback_period": {
        "train": [
            "What's the payback period for a {initial_investment} investment that returns {cash_flows}?",
            "How long until I recover {initial_investment} if the project brings in {cash_flows}?",
            "Payback period: cost {initial_investment}, cash flows {cash_flows}.",
            "Solar panels cost {initial_investment} and save me {annual_cash_flow} a year. How long until they pay for themselves?",
            "When do I break even on a {initial_investment} purchase that generates {cash_flows}?",
            "How many years to earn back {initial_investment} from {cash_flows}?",
            "A {initial_investment} upgrade saves {annual_cash_flow} per year. Payback period?",
            "We spent {initial_investment} on new machinery. It should generate {cash_flows}. What's the payback period?",
            "payback on {initial_investment} with {annual_cash_flow}/yr in savings",
        ],
        "test": [
            "Find how long it takes for cumulative cash flows of {cash_flows} to cover an initial cost of {initial_investment}.",
            "How quickly does a {initial_investment} upgrade pay for itself if it saves {annual_cash_flow} annually?",
            "Payback time on an investment of {initial_investment} with inflows of {cash_flows}?",
        ],
        "partial": [
            ("What's the payback period for a {initial_investment} investment?", ["cash_flows"]),
        ],
    },
    "dividend_yield": {
        "train": [
            "A stock trades at {share_price} and pays an annual dividend of {dividend_per_share}. What's the dividend yield?",
            "What's the dividend yield if the share price is {share_price} and it pays {dividend_per_share} {payments_per_year:adv}?",
            "Dividend yield: price {share_price}, dividend {dividend_per_share} per share per year.",
            "My stock pays a {payments_per_year:adj} dividend of {dividend_per_share} and costs {share_price}. What yield is that?",
            "If a company pays {dividend_per_share} a share each year and the stock is at {share_price}, what's the yield?",
            "{share_price} stock, {dividend_per_share} {payments_per_year:adj} dividend. Yield?",
            "How much yield am I getting from a stock priced at {share_price} with a {dividend_per_share} annual dividend?",
            "This REIT pays {dividend_per_share} {payments_per_year:adv} and trades for {share_price}. What's its annual dividend yield?",
        ],
        "test": [
            "Compute the dividend yield for shares costing {share_price} that pay {dividend_per_share} {payments_per_year:adv}.",
            "With a {dividend_per_share} yearly dividend and a {share_price} share price, what's the yield?",
            "A {payments_per_year:adj} payout of {dividend_per_share} on a {share_price} stock is what yield?",
        ],
        "partial": [
            ("A stock pays a {dividend_per_share} annual dividend. What's the dividend yield?", ["share_price"]),
        ],
    },
    "pe_ratio": {
        "train": [
            "What's the P/E ratio of a stock priced at {share_price} with earnings per share of {earnings_per_share}?",
            "Price {share_price}, EPS {earnings_per_share}. P/E?",
            "A company earns {earnings_per_share} per share and its stock trades at {share_price}. What's the price-to-earnings ratio?",
            "Calculate the PE for a {share_price} stock that earned {earnings_per_share} a share last year.",
            "How expensive is a stock at {share_price} if EPS is {earnings_per_share}? Give me the P/E.",
            "What's the earnings multiple for shares at {share_price} with {earnings_per_share} in EPS?",
            "pe ratio {share_price} {earnings_per_share}",
            "If earnings per share are {earnings_per_share} and the share price is {share_price}, what is the price/earnings ratio?",
        ],
        "test": [
            "Find the price-to-earnings ratio: share price {share_price}, earnings per share {earnings_per_share}.",
            "A stock at {share_price} with trailing EPS of {earnings_per_share} has what P/E multiple?",
            "How many times earnings is a {share_price} stock trading at if it earns {earnings_per_share} per share?",
        ],
        "partial": [
            ("What's the P/E ratio of a stock priced at {share_price}?", ["earnings_per_share"]),
        ],
    },
    "sharpe_ratio": {
        "train": [
            "What's the Sharpe ratio for a portfolio returning {portfolio_return} with {volatility} volatility if the risk-free rate is {risk_free_rate}?",
            "Sharpe ratio: return {portfolio_return}, risk-free {risk_free_rate}, standard deviation {volatility}.",
            "My fund returned {portfolio_return} last year with a standard deviation of {volatility}. T-bills paid {risk_free_rate}. Sharpe ratio?",
            "Calculate risk-adjusted return (Sharpe) for a {portfolio_return} return, {volatility} std dev, {risk_free_rate} risk-free rate.",
            "How good is a {portfolio_return} return with {volatility} volatility when the risk-free rate is {risk_free_rate}? Give me the Sharpe ratio.",
            "sharpe: rp {portfolio_return}, rf {risk_free_rate}, sigma {volatility}",
            "If my strategy averages {portfolio_return} a year, its volatility is {volatility}, and treasuries yield {risk_free_rate}, what's the Sharpe ratio?",
        ],
        "test": [
            "Compute the Sharpe ratio given an expected return of {portfolio_return}, a risk-free rate of {risk_free_rate} and volatility of {volatility}.",
            "With returns of {portfolio_return}, standard deviation {volatility}, and a {risk_free_rate} risk-free rate, what's the Sharpe?",
            "What's my portfolio's Sharpe ratio: {portfolio_return} annual return, {volatility} annualized volatility, {risk_free_rate} cash rate?",
        ],
        "partial": [
            ("What's the Sharpe ratio for a portfolio returning {portfolio_return} with {volatility} volatility?", ["risk_free_rate"]),
        ],
    },
    "average_cost_per_share": {
        "train": [
            "I invested {amount_per_purchase} each time a stock was at {share_prices}. What's my average cost per share?",
            "Dollar-cost averaging: {amount_per_purchase} per purchase at prices {share_prices}. Average cost?",
            "If I buy {amount_per_purchase} worth of an ETF each month and the prices were {share_prices}, how many shares do I have and what did I pay per share on average?",
            "What's my cost basis per share if I put {amount_per_purchase} in at each of these prices: {share_prices}?",
            "I've been buying {amount_per_purchase} of bitcoin every paycheck at {share_prices}. Average purchase price?",
            "avg cost per share: {amount_per_purchase} each buy, prices {share_prices}",
            "Monthly purchases of {amount_per_purchase} at {share_prices}. What's the average price I paid?",
        ],
        "test": [
            "With fixed investments of {amount_per_purchase} at share prices of {share_prices}, what's my average cost per share?",
            "I DCA {amount_per_purchase} per buy. Prices were {share_prices}. Total shares and average cost?",
            "Find the average cost basis when investing {amount_per_purchase} at each of the following prices: {share_prices}.",
        ],
        "partial": [
            ("I invested {amount_per_purchase} every month in an index fund. What's my average cost per share?", ["share_prices"]),
        ],
    },
    "investment_fee_impact": {
        "train": [
            "How much will a {annual_fee} annual fee cost me on {initial_investment} over {years} if the market returns {annual_return}?",
            "Compare {initial_investment} growing at {annual_return} for {years} with and without a {annual_fee} expense ratio.",
            "My advisor charges {annual_fee} a year. On a {initial_investment} portfolio earning {annual_return}, what's the total fee drag over {years}?",
            "Fee impact: {initial_investment}, {annual_return} return, {annual_fee} fee, {years}.",
            "How much less will I have after {years} on {initial_investment} if my fund charges {annual_fee} and returns {annual_return} before fees?",
            "Is a {annual_fee} fee a big deal? I have {initial_investment} invested earning {annual_return} for {years}.",
            "What does a {annual_fee} expense ratio cost over {years} on {initial_investment} at {annual_return}?",
        ],
        "test": [
            "Show the long-term cost of {annual_fee} in annual fees on {initial_investment} with {annual_return} gross returns over {years}.",
            "How much would I lose to a {annual_fee} management fee over {years} if {initial_investment} grows {annual_return} a year?",
            "Over {years}, how much do fees of {annual_fee} per year eat into {initial_investment} earning {annual_return}?",
        ],
        "partial": [
            ("How much will a {annual_fee} annual fee cost me over {years}?", ["initial_investment", "annual_return"]),
        ],
    },
    "savings_goal_monthly": {
        "train": [
            "How much do I need to save each month to reach {target_amount} in {years} at {annual_rate}?",
            "I want {target_amount} for a down payment in {years}. If my savings earn {annual_rate}, what's my monthly savings target?",
            "I have {current_savings} saved. How much more per month do I need to hit {target_amount} in {years} with a {annual_rate} return?",
            "Monthly contribution needed to reach {target_amount} in {years}, assuming {annual_rate} growth?",
            "To retire with {target_amount} in {years}, earning {annual_rate}, how much should I invest monthly? I'm starting with {current_savings}.",
            "What do I need to put away each month for {years} to end up with {target_amount}, at {annual_rate} interest?",
            "goal {target_amount} in {years}, {annual_rate} return, already have {current_savings}. monthly savings?",
            "We want {target_amount} saved for our kid's college in {years}. At {annual_rate}, how much per month?",
            "How much a month to build a {target_amount} emergency fund in {years} if the account pays {annual_rate}?",
        ],
        "test": [
            "What monthly deposit will grow to {target_amount} over {years} at {annual_rate}?",
            "Starting from {current_savings}, how much do I need to add every month to reach {target_amount} in {years} assuming {annual_rate} returns?",
            "I'm saving for a {target_amount} wedding in {years}. With {annual_rate} interest, what should I set aside monthly?",
        ],
        "partial": [
            ("How much do I need to save each month to reach {target_amount}?", ["years", "annual_rate"]),
            ("How much should I save monthly to reach {target_amount} in {years}?", ["annual_rate"]),
        ],
    },
    "retirement_nest_egg": {
        "train": [
            "How much do I need to retire if I spend {annual_expenses} a year?",
            "Using the 4% rule, how big a nest egg do I need for {annual_expenses} of annual expenses?",
            "I want to withdraw {withdrawal_rate} a year in retirement and live on {annual_expenses}. How much should I have saved?",
            "What's my FIRE number if my annual spending is {annual_expenses}?",
            "Retirement target for {annual_expenses} per year at a {withdrawal_rate} safe withdrawal rate?",
            "If I need {annual_expenses} a year to live on, how much money do I need invested to retire?",
            "nest egg needed for {annual_expenses}/yr spending, {withdrawal_rate} withdrawal rate",
            "We need {monthly_expenses} a month in retirement. How much should we have saved?",
            "If I spend {monthly_expenses} per month, what's my FIRE number at a {withdrawal_rate} withdrawal rate?",
        ],
        "test": [
            "What size portfolio supports {annual_expenses} in yearly withdrawals at a {withdrawal_rate} withdrawal rate?",
            "If my retirement budget is {monthly_expenses} a month, what's my target savings number?",
            "Calculate my financial independence number with annual expenses of {annual_expenses}.",
        ],
        "partial": [
            ("How much do I need saved to retire?", ["annual_expenses"]),
        ],
    },
    "retirement_drawdown": {
        "train": [
            "How long will {balance} last if I withdraw {annual_withdrawal} a year and earn {annual_return}?",
            "I'm retiring with {balance}. If I take out {monthly_withdrawal} a month and my investments return {annual_return}, when will I run out?",
            "How many years will my savings of {balance} last withdrawing {annual_withdrawal} annually at a {annual_return} return?",
            "Drawdown: {balance} portfolio, {annual_withdrawal}/year withdrawals, {annual_return} growth. Years until empty?",
            "Will {balance} be enough? I plan to spend {annual_withdrawal} a year from it and expect {annual_return} returns.",
            "If I live off {monthly_withdrawal} a month from a {balance} retirement account earning {annual_return}, how long does the money last?",
            "how long does {balance} last at {annual_withdrawal} per year, {annual_return} return",
            "My mom has {balance} saved and needs {monthly_withdrawal} per month. Assuming {annual_return} a year, how many years will it cover?",
        ],
        "test": [
            "Starting with {balance} and withdrawing {annual_withdrawal} each year, with {annual_return} annual returns, when does the balance hit zero?",
            "How long can I sustain {monthly_withdrawal} monthly withdrawals from {balance} if it earns {annual_return}?",
            "Longevity of a {balance} nest egg with yearly withdrawals of {annual_withdrawal} and {annual_return} growth?",
        ],
        "partial": [
            ("How long will {balance} last if I withdraw {annual_withdrawal} a year?", ["annual_return"]),
        ],
    },
    "emergency_fund": {
        "train": [
            "How big should my emergency fund be if my monthly expenses are {monthly_expenses}?",
            "I spend {monthly_expenses} a month. How much do I need for a {months:n}-month emergency fund?",
            "My expenses are {monthly_expenses} per month and I have {current_savings} saved. How far am I from a {months:n}-month emergency fund?",
            "Emergency fund target: {monthly_expenses}/month expenses, {months} of coverage.",
            "How much cash should I keep for emergencies to cover {months} of {monthly_expenses} expenses?",
            "I've got {current_savings} in savings and my bills run {monthly_expenses} a month. Do I have enough for {months} of emergencies?",
            "emergency fund for {monthly_expenses} monthly costs",
        ],
        "test": [
            "What emergency fund would cover {months} of living expenses at {monthly_expenses} a month?",
            "With {monthly_expenses} in monthly costs, how much should be in my rainy-day fund? I currently have {current_savings}.",
            "How large an emergency cushion do I need if I spend {monthly_expenses} monthly?",
        ],
        "partial": [
            ("How big should my emergency fund be?", ["monthly_expenses"]),
        ],
    },
    "debt_to_income": {
        "train": [
            "What's my debt-to-income ratio if I pay {monthly_debt_payments} a month in debts and earn {gross_monthly_income} a month before taxes?",
            "My monthly debt payments are {monthly_debt_payments} and my gross salary is {gross_annual_income} a year. What's my DTI?",
            "DTI: debts {monthly_debt_payments}/month, gross income {gross_monthly_income}/month.",
            "I make {gross_annual_income} a year and have {monthly_debt_payments} in monthly payments (car, student loans, credit cards). Will lenders think my DTI is too high?",
            "Calculate debt-to-income: {monthly_debt_payments} monthly obligations, {gross_monthly_income} monthly gross income.",
            "Can I qualify for a mortgage? I earn {gross_annual_income} annually and my monthly debts total {monthly_debt_payments}. What's my debt to income?",
            "dti {monthly_debt_payments} debt, {gross_monthly_income} income monthly",
        ],
        "test": [
            "With {monthly_debt_payments} of monthly debt payments and a {gross_annual_income} annual salary, what is my debt-to-income ratio?",
            "Monthly debts: {monthly_debt_payments}. Monthly gross pay: {gross_monthly_income}. DTI?",
            "What percent of my {gross_monthly_income} monthly income goes to my {monthly_debt_payments} in debt payments?",
        ],
        "partial": [
            ("What's my debt-to-income ratio if I pay {monthly_debt_payments} a month in debts?", ["gross_monthly_income"]),
        ],
    },
    "budget_50_30_20": {
        "train": [
            "How should I split {monthly_income} a month using the 50/30/20 rule?",
            "I take home {annual_income} a year. What does a 50/30/20 budget look like for me?",
            "50/30/20 budget for {monthly_income} monthly take-home pay?",
            "My paycheck after taxes is {monthly_income} a month. How much should go to needs, wants, and savings?",
            "Make me a simple budget with the 50 30 20 rule. I net {annual_income} annually.",
            "How much should I spend on needs vs wants vs savings if I bring home {monthly_income} per month?",
            "50-30-20 on {monthly_income}",
        ],
        "test": [
            "Using the 50/30/20 guideline, how would I allocate a monthly take-home of {monthly_income}?",
            "With {annual_income} in after-tax income per year, what are my 50/30/20 budget amounts?",
            "Break down {monthly_income} of net monthly income into needs, wants and savings.",
        ],
        "partial": [
            ("Can you make me a 50/30/20 budget?", ["monthly_income"]),
        ],
    },
    "break_even_units": {
        "train": [
            "How many units do I need to sell to break even if fixed costs are {fixed_costs}, the price is {price_per_unit} and variable cost is {variable_cost_per_unit} per unit?",
            "Break-even point: fixed costs {fixed_costs}, selling price {price_per_unit}, variable cost {variable_cost_per_unit}.",
            "My bakery has {fixed_costs} in monthly fixed costs. Each cake sells for {price_per_unit} and costs {variable_cost_per_unit} to make. How many cakes to break even?",
            "I'm launching a product priced at {price_per_unit} that costs {variable_cost_per_unit} per unit to produce. Fixed costs are {fixed_costs}. What's the break-even volume?",
            "How many t-shirts at {price_per_unit} each do I need to sell to cover {fixed_costs} of fixed costs if each shirt costs me {variable_cost_per_unit}?",
            "break even units: FC {fixed_costs}, P {price_per_unit}, VC {variable_cost_per_unit}",
            "What sales volume covers {fixed_costs} in overhead when each unit sells for {price_per_unit} and has {variable_cost_per_unit} in variable costs?",
            "We sell subscriptions for {price_per_unit} with {variable_cost_per_unit} in costs per subscriber and {fixed_costs} in fixed expenses. How many subscribers do we need to break even?",
        ],
        "test": [
            "Find the break-even quantity: {fixed_costs} fixed, {price_per_unit} per unit revenue, {variable_cost_per_unit} per unit variable cost.",
            "My food truck's fixed costs are {fixed_costs}. Meals sell for {price_per_unit} and cost {variable_cost_per_unit} each. How many meals until I break even?",
            "How many units must be sold at {price_per_unit} to cover fixed costs of {fixed_costs} with a {variable_cost_per_unit} variable cost each?",
        ],
        "partial": [
            ("How many units do I need to sell to break even if fixed costs are {fixed_costs}?", ["price_per_unit", "variable_cost_per_unit"]),
        ],
    },
    "profit_margin": {
        "train": [
            "What's my profit margin if revenue is {revenue} and costs are {cost}?",
            "Profit margin: sales {revenue}, total costs {cost}.",
            "My business made {revenue} last year and spent {cost}. What's the profit margin?",
            "Calculate margin for {revenue} in revenue and {cost} in expenses.",
            "We brought in {revenue} and our costs were {cost}. What percentage did we keep as profit?",
            "profit margin {revenue} rev {cost} cost",
            "If a project generated {revenue} and cost {cost} to deliver, what's the margin?",
        ],
        "test": [
            "With sales of {revenue} and costs of {cost}, what is the profit and profit margin?",
            "What share of {revenue} in revenue is profit if expenses were {cost}?",
            "Find the margin for a quarter with {revenue} of revenue and {cost} of costs.",
        ],
        "partial": [
            ("What's my profit margin if revenue is {revenue}?", ["cost"]),
        ],
    },
    "markup_percentage": {
        "train": [
            "What's the markup if I buy something for {cost} and sell it for {selling_price}?",
            "Markup percentage: cost {cost}, selling price {selling_price}.",
            "I buy sneakers wholesale at {cost} and sell them at {selling_price}. What's my markup and margin?",
            "If an item costs me {cost} and I price it at {selling_price}, what markup is that?",
            "How much markup on a {cost} item selling for {selling_price}?",
            "markup {cost} -> {selling_price}",
            "I make candles for {cost} each and sell them for {selling_price}. What's my markup on cost?",
        ],
        "test": [
            "Find the markup and gross margin for a product with a cost of {cost} and a price of {selling_price}.",
            "Selling at {selling_price} something that costs {cost}: what's the percentage markup?",
            "What markup did I apply if my unit cost is {cost} and the retail price is {selling_price}?",
        ],
        "partial": [
            ("What's the markup if I buy something for {cost}?", ["selling_price"]),
        ],
    },
    "straight_line_depreciation": {
        "train": [
            "What's the annual depreciation on a {asset_cost} machine with a {salvage_value} salvage value and a {useful_life_years} useful life?",
            "Straight-line depreciation: cost {asset_cost}, salvage {salvage_value}, life {useful_life_years}.",
            "We bought a delivery van for {asset_cost}. It'll be worth {salvage_value} after {useful_life_years}. How much does it depreciate per year?",
            "Depreciate {asset_cost} of equipment over {useful_life_years} with no salvage value using straight line.",
            "How much depreciation per year for a {asset_cost} asset lasting {useful_life_years} with {salvage_value} residual value?",
            "straight line dep {asset_cost} cost {salvage_value} salvage {useful_life_years:n} yrs",
            "My company's new server cost {asset_cost} and has a useful life of {useful_life_years}. What's the yearly straight-line depreciation if salvage is {salvage_value}?",
        ],
        "test": [
            "Compute yearly straight-line depreciation for an asset costing {asset_cost}, salvage {salvage_value}, useful life {useful_life_years}.",
            "A {asset_cost} tractor will be sold for {salvage_value} after {useful_life_years}. What's the annual depreciation expense?",
            "What's the per-year depreciation on {asset_cost} of office furniture over {useful_life_years}, salvage value {salvage_value}?",
        ],
        "partial": [
            ("What's the annual depreciation on a {asset_cost} machine?", ["useful_life_years"]),
        ],
    },
    "percentage_change": {
        "train": [
            "What's the percentage change from {old_value} to {new_value}?",
            "My rent went from {old_value:money} to {new_value:money}. What percent increase is that?",
            "Our website traffic went from {old_value} visits to {new_value}. What's the percent change?",
            "Percent change: old {old_value}, new {new_value}.",
            "My salary changed from {old_value:money} to {new_value:money}. By what percentage?",
            "The price of eggs went from {old_value:money} to {new_value:money}. How much is that in percent?",
            "{old_value} to {new_value} percent change?",
            "Sales were {old_value:money} last quarter and {new_value:money} this quarter. What's the percentage growth?",
        ],
        "test": [
            "How big is the percentage increase or decrease going from {old_value} to {new_value}?",
            "My electric bill changed from {old_value:money} to {new_value:money}. What's the percent change?",
            "Followers went from {old_value} to {new_value}. Percent change?",
        ],
        "partial": [
            ("My rent went up to {new_value:money}. What percent increase is that?", ["old_value"]),
        ],
    },
}
