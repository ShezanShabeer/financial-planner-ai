import random, math
from pathlib import Path

UAELocs = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain", "Damac Hills", "Dubai Marina", "JLT", "Mirdif", "Al Barsha", "Deira"]
Currencies = ["AED"]  # keep AED dominant; model still learns symbols/words later
Weddings = ["My daughter's wedding", "My son's wedding", "Our wedding", "daughters wedding", "son's wedding", "My daughter's marriage", "My son's marriage", "Our marriage"]
Vehicles = ["car", "SUV", "bike", "toyota", "nissan"]
Luxury = ["Lamborghini", "Rolex", "Ferrari"]
EduPhrases = ["college", "university", "school"]
Business = ["restaurant business", "trading company", "cafe", "startup", "restaurant", "hotel"]
Travel = ["travel the world", "long trip to Europe", "visit 10 countries"]

def k(v): return f"{v}k"
def m(v): return f"{v}m"

def rand_amount():
    r = random.random()
    if r < 0.5:
        return f"{random.choice([50,60,70,80,100,120,150,200,300])}k"
    elif r < 0.8:
        return f"{random.choice([1,1.5,2,2.5,3])}m"
    else:
        return f"{random.choice([50000,80000,120000,150000,200000,300000,1000000])}"

def rand_years():
    return random.choice([1,2,3,4,5,7,10,15,20])

def one_house():
    amt = rand_amount()
    yr = rand_years()
    loc = random.choice(UAELocs)
    return f"I want to save {amt} {random.choice(Currencies)} in {yr} years to buy a house in {loc}."

def one_wedding():
    amt = rand_amount()
    yr = random.choice([1,2,3,4,5])
    who = random.choice(Weddings)
    return f"{who} is planned in {yr} years. I need about {amt} AED for it."

def one_vehicle():
    amt = rand_amount()
    when = random.choice(["next year","in 2 years","in 6 months"])
    what = random.choice(Vehicles)
    return f"I want to buy a {what} worth {amt} AED {when}."

def one_business():
    amt = rand_amount()
    yr = random.choice([2,3,5])
    loc = random.choice(UAELocs)
    return f"I want to start a {random.choice(Business)} in {loc} in {yr} years which would need around {amt} AED."

def one_retirement():
    yr = random.choice([10,15,20,25])
    amt = random.choice(["1m", "2m", "3m", "1,000,000"])
    return f"I want to retire in {yr} years with {amt} AED."

def one_realestate_portfolio():
    loc = random.choice(UAELocs)
    return f"I want to build a real estate investment portfolio in {loc} and live off the rental income."

def one_edu():
    yr = random.choice([1,2,3])
    fee = random.choice([15000,25000,50000])
    return f"My child will start {random.choice(EduPhrases)} in {yr} years; the fee is {fee} AED per year for 4 years."

def one_travel():
    yr = random.choice([2,3,5,10])
    return f"I want to quit my job and {random.choice(Travel)} in {yr} years."

def one_luxury():
    yr = random.choice([3,5,7])
    item = random.choice(Luxury)
    return f"I want to buy a {item} in {yr} years."

def one_gift():
    loc = random.choice(["Dubai Marina","Downtown Dubai","JBR"])
    return f"I want to gift my spouse a 2BHK apartment in {loc}."

GENS = [one_house, one_wedding, one_vehicle, one_business, one_retirement, one_realestate_portfolio, one_edu, one_travel, one_luxury, one_gift]

def main(n=250, out_path="data/nlp/sentences.txt", seed=42):
    random.seed(seed)
    lines = []
    for _ in range(n):
        lines.append(random.choice(GENS)())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} sentences to {out_path}. Review and add real samples too!")

if __name__ == "__main__":
    main()
