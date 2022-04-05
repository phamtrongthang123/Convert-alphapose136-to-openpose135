res = set()
for a in range(0,57):
    for b in range(0,57):
        for c in range(0,57):
            if a+b+c > 56:
                continue 
            d = 56 - (a+b+c)
            score = (a+b*2+c*3+d*4) / 56
            res.add(round(score,1))

res = sorted(res)
print(res)