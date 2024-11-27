import sys
input = sys.stdin.readline

n = int(input())

arr = []

for i in range(n):
    arr.append(int(input()))

arr.sort()

result = arr[0]
tmp = arr[0]
cnt = 1
choibin = []

for i in range(1, n):
    result += arr[i]
    if arr[i] == tmp:
        cnt += 1
    else:
        choibin.append([tmp, cnt])
        tmp = arr[i]
        cnt = 1

choibin.append([tmp, cnt])

print(round(result/n))
print(arr[n//2])

choibin.sort(key= lambda x : (x[1], -x[0]))
if len(choibin) >= 2:
    if choibin[-1][1] == choibin[-2][1]:
        print(choibin[-2][0])
    else:
        print(choibin[-1][0])
else:
    print(choibin[-1][0])

print(arr[-1] - arr[0])

