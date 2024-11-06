class Solution {
public:
    bool isCircularSentence(string sentence) {
        auto ss = split(sentence, ' ');
        int n = ss.size();
        for (int i = 0; i < n; ++i) {
            if (ss[i].back() != ss[(i + 1) % n][0]) {
                return false;
            }
        }
        return true;
    }

class Solution {
public:
    bool canSortArray(vector<int>& nums) {
        int preMx = 0;
        int i = 0, n = nums.size();
        while (i < n) {
            int cnt = __builtin_popcount(nums[i]);
            int j = i + 1;
            int mi = nums[i], mx = nums[i];
            while (j < n && __builtin_popcount(nums[j]) == cnt) {
                mi = min(mi, nums[j]);
                mx = max(mx, nums[j]);
                j++;
            }
            if (preMx > mi) {
                return false;
            }
            preMx = mx;
            i = j;
        }
        return true;
    }
};
