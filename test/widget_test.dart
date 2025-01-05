class Solution {
public:
    int maximumBeauty(vector<int>& nums, int k) {
        int m = *max_element(nums.begin(), nums.end()) + k * 2 + 2;
        vector<int> d(m);
        for (int x : nums) {
            d[x]++;
            d[x + k * 2 + 1]--;class Solution class Solution {
    public int countPalindromicSubsequence(String s) {
        int ans = 0;
        for (char c = 'a'; c <= 'z'; ++c) {
            int l = s.indexOf(c), r = s.lastIndexOf(c);
            Set<Character> cs = new HashSet<>();
            for (int i = l + 1; i < r; ++i) {
                cs.add(s.charAt(i));
            }
            ans += cs.size();
        }class Solution {
    public String shiftingLetters(String s, int[][] shifts) {
        int n = s.length();
        int[] d = new int[n + 1];
        for (int[] e : shifts) {
            if (e[2] == 0) {
                e[2]--;
            }
            d[e[0]] += e[2];
            d[e[1] + 1] -= e[2];
        }
        for (int i = 1; i <= n; ++i) {
            d[i] += d[i - 1];
        }
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < n; ++i) {
            int j = (s.charAt(i) - 'a' + d[i] % 26 + 26) % 26;
            ans.append((char) ('a' + j));
        }
        return ans.toString();
    }
}
        return ans;
    }
}{
    public int maxScore(String s) {
        int l = 0, r = 0;
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == '1') {
                ++r;
                class Solution {
    public int waysToSplitArray(int[] nums) {
        long s = 0;
        for (int x : nums) {
            s += x;
        }
        long t = 0;
        int ans = 0;
        for (int i = 0; i + 1 < nums.length; ++i) {
            t += nums[i];
            ans += t >= s - t ? 1 : 0;
        }
        return ans;
    }
}
            }class Solution {
    private List<Integer> nums = new ArrayList<>();

    public int[] vowelStrings(String[] words, int[][] queries) {
        Set<Character> vowels = Set.of('a', 'e', 'i', 'o', 'u');
        for (int i = 0; i < words.length; ++i) {
            char a = words[i].charAt(0), b = words[i].charAt(words[i].length() - 1);
            if (vowels.contains(a) && vowels.contains(b)) {
                nums.add(i);
            }
        }
        int m = queries.length;
        int[] ans = new int[m];
        for (int i = 0; i < m; ++i) {
            int l = queries[i][0], r = queries[i][1];
            ans[i] = search(r + 1) - search(l);
        }
        return ans;
    }

    private int search(int x) {
        int l = 0, r = nums.size();
        while (l < r) {
            int mid = (l + r) >> 1;
            if (nums.get(mid) >= x) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }
}
        }
        int ans = 0;
        for (int i = 0; i < n - 1; ++i) {
            l += (s.charAt(i) - '0') ^ 1;
            r -= s.charAt(i) - '0';
            ans = Math.max(ans, l + r);
        }
        return ans;
    }
}
            class Solution {
                class Solution {
                    class Solution {class Solution:
                                    class Solution {
    private final int[] valid = {1, 7, 30};
    private int[] days;
    private int[] costs;
    private Integer[] f;
    private int n;

    public int mincostTickets(int[] days, int[] costs) {
        n = days.length;
        f = new Integer[n];
        this.days = days;
        this.costs = costs;
        return dfs(0);
    }

    private int dfs(int i) {
        if (i >= n) {
            return 0;
        }
        if (f[i] != null) {
            return f[i];
        }
        f[i] = Integer.MAX_VALUE;
        for (int k = 0; k < 3; ++k) {
            int j = Arrays.binarySearch(days, days[i] + valid[k]);
            j = j < 0 ? -j - 1 : j;
            f[i] = Math.min(f[i], dfs(j) + costs[k]);
        }
        return f[i];
    }
}
    def numWays(self, words: List[str], target: str) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i >= m:
                return 1
            if j >= n:
                return 0
            ans = dfs(i + 1, j + 1) * cnt[j][ord(target[i]) - ord('a')]
            ans = (ans + dfs(i, j + 1)) % mod
            return ans

        m, n = len(target), len(words[0])
        cnt = [[0] * 26 for _ in range(n)]
        for w in words:
            for j, c in enumerate(w):
                cnt[j][ord(c) - ord('a')] += 1
        mod = 10**9 + 7
        return dfs(0, 0)
public:
    int maxScoreSightseeingPair(vector<int>& values) {
        int ans = 0, mx = 0;
        for (int j = 0; j < values.size(); ++j) {
            ans = max(ans, mx + values[j] - j);
            mx = max(mx, values[j] + j);
        }
        return ans;
    }
};
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int s = accumulate(nums.begin(), nums.end(), 0);
        if (s < target || (s - target) % 2) {
            return 0;
        }
        int m = nums.size();
        int n = (s - target) / 2;
        int f[m + 1][n + 1];
        memset(f, 0, sizeof(f));
        f[0][0] = 1;
        for (int i = 1; i <= m; ++i) {
            for (int j = 0; j <= n; ++j) {
                f[i][j] = f[i - 1][j];
                if (j >= nums[i - 1]) {
                    f[i][j] += f[i - 1][j - nums[i - 1]];
                }
            }
        }
        return f[m][n];
    }
};3
                421.16	421.16
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> largestValues(TreeNode* root) {
        vector<int> ans;
        if (!root) {
            return ans;
        }
        queue<TreeNode*> q{{root}};
        while (q.size()) {
            int x = INT_MIN;
            for (int i = q.size(); i; --i) {
                TreeNode* node = q.front();
                q.pop();
                x = max(x, node->val);
                if (node->left) {
                    q.push(node->left);
                }
                if (node->right) {
                    q.push(node->right);
                }
            }
            ans.push_back(x);
        }
        return ans;
    }
};
public:
    int minimumDiameterAfterMerge(vector<vector<int>>& edges1, vector<vector<int>>& edges2) {
        int d1 = treeDiameter(edges1);
        int d2 = treeDiameter(edges2);
        return max({d1, d2, (d1 + 1) / 2 + (d2 + 1) / 2 + 1});
    }

    int treeDiameter(vector<vector<int>>& edges) {
        int n = edges.size() + 1;
        vector<int> g[n];
        for (auto& e : edges) {
            int a = e[0], b = e[1];
            g[a].push_back(b);
            g[b].push_back(a);
        }
        int ans = 0, a = 0;
        auto dfs = [&](this auto&& dfs, int i, int fa, int t) -> void {
            for (int j : g[i]) {
                if (j != fa) {
                    dfs(j, i, t + 1);
                }
            }
            if (ans < t) {
                ans = t;
                a = i;
            }
        };
        dfs(0, -1, 0);
        dfs(a, -1, 0);
        return ans;
    }
};
        }class Solution {
public:
            class BinaryIndexedTree {
private:/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int minimumOperations(TreeNode* root) {
        queue<TreeNode*> q{{root}};
        int ans = 0;
        auto f = [](vector<int>& t) {
            int n = t.size();
            vector<int> alls(t.begin(), t.end());
            sort(alls.begin(), alls.end());
            unordered_map<int, int> m;
            int ans = 0;
            for (int i = 0; i < n; ++i) m[alls[i]] = i;
            for (int i = 0; i < n; ++i) t[i] = m[t[i]];
            for (int i = 0; i < n; ++i) {
                while (t[i] != i) {
                    swap(t[i], t[t[i]]);
                    ++ans;
                }
            }
            return ans;
        };
        while (!q.empty()) {
            vector<int> t;
            for (int n = q.size(); n; --n) {
                auto node = q.front();
                q.pop();
                t.emplace_back(node->val);
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            ans += f(t);
        }
        return ans;
    }
};
    int inf = 1 << 30;
    int n;
    vector<int> c;

public:
    BinaryIndexedTree(int n) {
        this->n = n;
        c.resize(n + 1, inf);
    }

    void update(int x, int v) {
        while (x <= n) {
            c[x] = min(c[x], v);
            x += x & -x;
        }
    }

    int query(int x) {
        int mi = inf;
        while (x > 0) {
            mi = min(mi, c[x]);
            x -= x & -x;
        }
        return mi == inf ? -1 : mi;
    }
};

class Solution {
public:
    vector<int> leftmostBuildingQueries(vector<int>& heights, vector<vector<int>>& queries) {
        int n = heights.size(), m = queries.size();
        for (auto& q : queries) {
            if (q[0] > q[1]) {
                swap(q[0], q[1]);
            }
        }
        vector<int> idx(m);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int i, int j) {
            return queries[j][1] < queries[i][1];
        });
        vector<int> s = heights;
        sort(s.begin(), s.end());
        s.erase(unique(s.begin(), s.end()), s.end());
        vector<int> ans(m);
        int j = n - 1;
        BinaryIndexedTree tree(n);
        for (int i : idx) {
            int l = queries[i][0], r = queries[i][1];
            while (j > r) {
                int k = s.end() - lower_bound(s.begin(), s.end(), heights[j]) + 1;
                tree.update(k, j);
                --j;
            }
            if (l == r || heights[l] < heights[r]) {
                ans[i] = r;
            } else {
                int k = s.end() - lower_bound(s.begin(), s.end(), heights[l]);
                ans[i] = tree.query(k);
            }
        }
        return ans;
    }
};
    int maxKDivisibleComponents(int n, vector<vector<int>>& edges, vector<int>& values, int k) {
        int ans = 0;
        vector<int> g[n];
        for (auto& e : edges) {
            int a = e[0], b = e[1];
            g[a].push_back(b);
            g[b].push_back(a);
        }
        function<long long(int, int)> dfs = [&](int i, int fa) {
            long long s = values[i];
            for (int j : g[i]) {
                if (j != fa) {
                    s += dfs(j, i);
                }
            }
            ans += s % k == 0;
            return s;
        };
        dfs(0, -1);
        return ans;
    }
};
        int ans = 0, s = 0;
        for (int x : d) {
            s += x;
            ans = max(ans, s);
        }
        }class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        int ans = 0, mx = 0;
        for (int i = 0; i < arr.size(); ++i) {
            mx = max(mx, arr[i]);
            ans += i == mx;
        }
        return ans;
    }.l
        
};
        class Solution {
public:
    vector<int> finalPrices(vector<int>& prices) {
        stack<int> stk;
        for (int i = prices.size() - 1; ~i; --i) {
            int x = prices[i];
            while (!stk.empty() && stk.top() > x) {
                stk.pop();
            }
            if (!stk.empty()) {
                prices[i] -= stk.top();
            }
            stk.push(x);
        }
        return prices;
    }
};
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
