class Solution {
public:
    vector<vector<char>> rotateTheBox(vector<vector<char>>& box) {
        int m = box.size(), n = box[0].size();
        vector<vector<char>> ans(n, vector<char>(m));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ans[j][m - i - 1] = box[i][j];
            }
class Solution {
public:
    long long maxMatrixSum(vector<vector<int>>& matrix) {
        long long s = 0;
        int cnt = 0, mi = INT_MAX;
        for (auto& row : matrix) {
            for (int& v : row) {
                s += abs(v);
                mi = min(mi, abs(v));
                cnt += v < 0;
            }
        }
        if (cnt % 2 == 0 || mi == 0) return s;
        return s - mi * 2;
    }
};
 
