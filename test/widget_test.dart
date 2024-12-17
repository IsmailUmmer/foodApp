class Solution {
public:
    int maximumBeauty(vector<int>& nums, int k) {
        int m = *max_element(nums.begin(), nums.end()) + k * 2 + 2;
        vector<int> d(m);
        for (int x : nums) {
            d[x]++;
            d[x + k * 2 + 1]--;
        }
        int ans = 0, s = 0;
        for (int x : d) {
            s += x;
            ans = max(ans, s);
        }
        }
        
        class Solution {
public:
    vector<int> getFinalState(vector<int>& nums, int k, int multiplier) {
        auto cmp = [&nums](int i, int j) {
            return nums[i] == nums[j] ? i > j : nums[i] > nums[j];
        };
        priority_queue<int, vector<int>, decltype(cmp)> pq(cmp);

        for (int i = 0; i < nums.size(); ++i) {
            pq.push(i);
        }

        while (k--) {
            int i = pq.top();
            pq.pop();
            nums[i] *= multiplier;
            pq.push(i);
        }

        return nums;
    }
};return ans;
    }
    class Solution {
public:
    double maxAverageRatio(vector<vector<int>>& classes, int extraStudents) {
        priority_queue<tuple<double, int, int>> pq;
        for (auto& e : classes) {
            int a = e[0], b = e[1];
            double x = (double) (a + 1) / (b + 1) - (double) a / b;
            pq.push({x, a, b});
        }
        while (extraStudents--) {
            auto [_, a, b] = pq.top();
            pq.pop();
            a++;
            b++;
            double x = (double) (a + 1) / (b + 1) - (double) a / b;
            pq.push({x, a, b});
        }
        double ans = 0;
        while (pq.size()) {
            auto [_, a, b] = pq.top();
            pq.pop();
            ans += (double) a / b;
        }
        return ans / classes.size();
    }
};
    class Solution {
public:
    long long continuousSubarrays(vector<int>& nums) {
        long long ans = 0;
        int i = 0, n = nums.size();
        multiset<int> s;
        for (int j = 0; j < n; ++j) {
            s.insert(nums[j]);
            while (*s.rbegin() - *s.begin() > 2) {
                s.erase(s.find(nums[i++]));
            }
            ans += j - i + 1;
        }
        return ans;
    }
};
};
class Solution {
public:
    long long findScore(vector<int>& nums) {
        int n = nums.size();
        vector<bool> vis(n);
        using pii = pair<int, int>;
        priority_queue<pii, vector<pii>, greater<pii>> q;
        for (int i = 0; i < n; ++i) {
            q.emplace(nums[i], i);
        }
        long long ans = 0;
        while (!q.empty()) {
            auto [x, i] = q.top();
            q.pop();
            ans += x;
            vis[i] = true;
            if (i + 1 < n) {
                vis[i + 1] = true;
            }
            if (i - 1 >= 0) {
                vis[i - 1] = true;
            }
            while (!q.empty() && vis[q.top().second]) {
                q.pop();
            }
        }
        return ans;
    }
};
