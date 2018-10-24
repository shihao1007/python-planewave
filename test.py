class Solution:
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        ans = [0] * (len(T) - 1)
        for i, t in enumerate(T[:-1]):
            cnt = 0
            for j in T[i+1:]:
                if j <= t:
                    cnt += 1
                else:
                    cnt += 1
                    break
            ans[i] = cnt if j > t else 0
        return ans + [0]
        

#print(openLock(deadends, target))

## Your MovingAverage object will be instantiated and called as such:
#rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]
T = [55,38,53,81,61,93,97,32,43,78]
obj = Solution
ans = obj.dailyTemperatures(obj, T)
#
#for i, row in enumerate(rooms):
#    print(i, row)
#    
#for j, r in enumerate(row):
#    print(j, r)