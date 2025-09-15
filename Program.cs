using System.ComponentModel;
using System.Diagnostics;
using Microsoft.Win32;



namespace FlashGameLoader
{
    public static class CaptchaPredictor
    {
        public static async Task<string?> Predict(string imagePath)
        {
            try
            {
                if (!File.Exists("predict.py"))
                {
                    MessageBox.Show("找不到 predict.py 檔案，程式即將關閉", "錯誤",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                    Application.Exit();
                }

                if (!File.Exists("best_mobilenet_captcha_model.pth"))
                {
                    MessageBox.Show("找不到模型檔案 best_mobilenet_captcha_model.pth，程式即將關閉", "錯誤",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                    Application.Exit();
                }

                return await Task.Run(() =>
                {
                    var psi = new ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = $"predict.py \"{imagePath}\" best_mobilenet_captcha_model.pth s",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };

                    using var process = Process.Start(psi)!;
                    string output = process.StandardOutput.ReadToEnd();
                    string error = process.StandardError.ReadToEnd();
                    process.WaitForExit();

                    if (process.ExitCode == 0 && !string.IsNullOrEmpty(output))
                    {
                        string result = output.Trim().Split('\n')[0].Trim();
                        if (result.Length == 4)
                            return result.ToUpper();
                    }

                    return null;
                });
            }
            catch
            {
                return null;
            }
        }

    }

    public class Program : Form
    {
        private WebBrowser webBrowser;
        private Label statusLabel;

        // 設定瀏覽器IE的版本為IE11，設定完才可以用CSS的方式修改網頁的縮放，預設的IE7沒辦法用CSS
        private static void SetBrowserFeatureControl()
        {
            string appName = System.IO.Path.GetFileName(Process.GetCurrentProcess().MainModule!.FileName);
            using (var key = Registry.CurrentUser.CreateSubKey(
                @"Software\Microsoft\Internet Explorer\Main\FeatureControl\FEATURE_BROWSER_EMULATION",
                RegistryKeyPermissionCheck.ReadWriteSubTree))
            {
                key.SetValue(appName, 11001, RegistryValueKind.DWord); // 11001 = IE11 edge mode
            }
        }

        [STAThread]
        public static void Main()
        {
            SetBrowserFeatureControl();
            Application.EnableVisualStyles();
            Application.Run(new Program());
        }

        // 介面主程式，網頁的邏輯，網頁的按鈕，網頁顯示文字
        public Program()
        {
            this.Text = "9s";
            this.Width = 1024;
            this.Height = 768;

            // 設定重新整理的按鈕和擷取圖片按鈕
            ToolStrip toolStrip = new ToolStrip();
            ToolStripButton refreshButton = new ToolStripButton("重新整理");
            ToolStripButton refreshCodeButton = new ToolStripButton("重新整理驗證碼");

            toolStrip.Items.Add(refreshButton);
            toolStrip.Items.Add(refreshCodeButton);

            // 顯示 PID 和狀態資訊
            Label pidLabel = new Label { AutoSize = true, Text = $"PID: {Process.GetCurrentProcess().Id}", Padding = new Padding(5) };
            statusLabel = new Label { AutoSize = true, Text = "準備中...", Padding = new Padding(5) };

            toolStrip.Items.Add(new ToolStripControlHost(pidLabel));
            toolStrip.Items.Add(new ToolStripControlHost(statusLabel));

            webBrowser = new WebBrowser
            {
                Dock = DockStyle.Fill,
                ScriptErrorsSuppressed = true // 隱藏指令碼錯誤視窗
            };

            // 綁定 NewWindow 事件
            webBrowser.NewWindow += WebBrowser_NewWindow;

            this.Controls.Add(webBrowser);
            this.Controls.Add(toolStrip);
            toolStrip.Dock = DockStyle.Top;

            // 載入遊戲網頁
            webBrowser.Navigate("https://member.9splay.com/Manage/Login?ReturnUrl=80764fb8749f652e58647354b2126cd171230cab4e57c01607214be84d78004f");

            // 綁定重新整理按鈕
            refreshButton.Click += (sender, e) =>
            {
                webBrowser.Refresh();
                // 關閉網頁離開提示
                webBrowser.Document!.InvokeScript("execScript", new object[]
                {
                    "window.onbeforeunload = null; window.onunload = null;"
                });
            };



            refreshCodeButton.Click += (sender, e) =>
            {
                RefreshVerifyCodeImage();
            };

            webBrowser.DocumentCompleted += (s, e) =>
            {
                if (webBrowser.Document != null && webBrowser.Url!.AbsoluteUri.Contains("/Login"))
                {
                    // 自動輸入帳號
                    var userBox = webBrowser.Document!.GetElementById("UserID");
                    if (userBox != null)
                    {
                        userBox.SetAttribute("value", "zxc11334342");
                    }
                    // 自動輸入密碼
                    var passBox = webBrowser.Document.GetElementById("UserPwd");
                    if (passBox != null)
                    {
                        passBox.SetAttribute("value", "zxc21735852");
                    }
                    // 自動擷取驗證碼圖片
                    CaptureVerifyCodeImage();
                }
                // 用JS把左邊的資訊欄刪除，不刪除會影響到畫面顯示
                if (webBrowser.Document != null && webBrowser.Url!.AbsoluteUri.Contains("/Game/Server/"))
                {
                    string script = @"
                        var nav = document.getElementById('nav');
                        if (nav) {
                            nav.parentNode.removeChild(nav);
                        }

                        // 刪除 id='btn_menu_close'
                        var btnClose = document.getElementById('btn_menu_close');
                        if (btnClose) {
                            btnClose.parentNode.removeChild(btnClose);
                        }

                        // 刪除 id='btn_menu_open'
                        var btnOpen = document.getElementById('btn_menu_open');
                        if (btnOpen) {
                            btnOpen.parentNode.removeChild(btnOpen);
                        };

                        var wap = document.querySelector('.game_wap');
                        if (wap) {
                            var tds = wap.querySelectorAll('td');
                            for (var i = 0; i < tds.length; i++) {
                                var td = tds[i];
                                if (td.querySelector('.game_left') || td.querySelector('.game_bar')) {
                                    td.parentNode.removeChild(td);
                                }
                            }
                        }
                    ";
                    webBrowser.Document.InvokeScript("eval", new object[] { script });

                    UpdateStatus($"遊戲進行中");
                    refreshCodeButton.Visible = false;
                }

                if (webBrowser.Document != null && webBrowser.Url!.AbsoluteUri == "http://san.9splay.com/")
                {
                    UpdateStatus("選擇伺服器");
                    webBrowser.Navigate("http://san.9splay.com/Game/Server/92");
                }
            };
        }

        // 更新狀態顯示
        private void UpdateStatus(string message)
        {
            if (statusLabel.InvokeRequired)
            {
                statusLabel.Invoke(new Action(() => statusLabel.Text = message));
            }
            else
            {
                statusLabel.Text = message;
            }
        }

        // 用JS擷取圖片並用python預測
        private async void CaptureVerifyCodeImage()
        {
            try
            {
                // 方法1：使用JavaScript Canvas方式擷取（推薦，品質最好）
                string script = @"
                    (function() {
                        var img = document.getElementById('verifyCodeImg');
                        if (!img) return '';
                        
                        // 等待圖片載入完成
                        if (!img.complete || img.naturalHeight === 0) {
                            return 'loading';
                        }
                        
                        var canvas = document.createElement('canvas');
                        canvas.width = img.naturalWidth || img.width;
                        canvas.height = img.naturalHeight || img.height;
                        
                        var ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0);
                        
                        return canvas.toDataURL('image/png');
                    })();
                ";

                object result = webBrowser.Document!.InvokeScript("eval", new object[] { script })!;
                string base64Data = result?.ToString()!;

                if (base64Data == "loading")
                {
                    UpdateStatus("圖片正在載入中，請稍後再試");
                    return;
                }

                if (!string.IsNullOrEmpty(base64Data) && base64Data.StartsWith("data:image"))
                {
                    // 移除data:image/png;base64,前綴
                    string base64String = base64Data.Substring(base64Data.IndexOf(',') + 1);
                    // 原本是先儲存圖片再讀取，之後才預測，現在直接用原本的編碼用python轉成圖片做預測
                    // byte[] imageBytes = Convert.FromBase64String(base64String);
                    // string folder = Path.Combine(Application.StartupPath, "captcha");

                    // // 如果資料夾不存在就建立
                    // if (!Directory.Exists(folder))
                    // {
                    //     Directory.CreateDirectory(folder);
                    // }

                    // // 儲存圖片
                    // string fileName = $"verifycode_{DateTime.Now:yyyyMMdd_HHmmss}.png";
                    // string filePath = Path.Combine(folder, fileName);
                    // File.WriteAllBytes(filePath, imageBytes);
                    // UpdateStatus($"驗證碼圖片已儲存至: {filePath}");


                    // 預測驗證碼並填入結果，await代表執行完才會執行後面的程式，如果要用圖片預測，把上面註解掉後將參數改成filePath
                    await PredictAndFillCaptcha(base64String);

                    // 只有在預測成功填入後才自動登入
                    var captchaInput = webBrowser.Document!.GetElementById("CheckText");
                    if (captchaInput != null && !string.IsNullOrEmpty(captchaInput.GetAttribute("value")))
                    {
                        ClickLogin();
                    }
                }

            }
            catch (Exception ex)
            {
                UpdateStatus($"擷取驗證碼圖片失敗: {ex.Message}");
            }
        }

        // 用python預測
        private async Task PredictAndFillCaptcha(string imagePath)
        {
            try
            {
                string? predictedResult = await CaptchaPredictor.Predict(imagePath)!;

                if (!string.IsNullOrEmpty(predictedResult))
                {
                    // 自動填入預測結果到CheckText欄位
                    var captchaInput = webBrowser.Document!.GetElementById("CheckText");
                    if (captchaInput != null)
                    {
                        captchaInput.SetAttribute("value", predictedResult);
                        UpdateStatus($"驗證碼識別結果: {predictedResult}");
                    }
                    else
                    {
                        UpdateStatus($"驗證碼識別成功: {predictedResult}，但找不到輸入框");
                    }
                }
                else
                {
                    UpdateStatus("python error");
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"驗證碼預測失敗: {ex.Message}");
            }
        }


        // 執行登入的JS
        private void ClickLogin()
        {
            webBrowser.Document!.InvokeScript("dosubmit");
        }


        // 重新整理驗證碼圖片
        private void RefreshVerifyCodeImage()
        {
            try
            {
                var img = webBrowser.Document?.GetElementById("verifyCodeImg");
                if (img != null)
                {
                    UpdateStatus("重新整理驗證碼中...");
                    // 模擬點擊圖片來重新整理驗證碼
                    img.InvokeMember("click");

                    // 延遲後自動擷取新的驗證碼
                    System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
                    timer.Interval = 500; // 延遲0.5秒等待新圖片載入
                    timer.Tick += (sender, e) =>
                    {
                        timer.Stop();
                        timer.Dispose();
                        CaptureVerifyCodeImage();
                    };
                    timer.Start();
                }
                else
                {
                    UpdateStatus("找不到驗證碼圖片");
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"重新整理失敗: {ex.Message}");
            }
        }

        // 攔截新視窗事件，強制在同一個 WebBrowser 開啟
        private void WebBrowser_NewWindow(object? sender, CancelEventArgs e)
        {
            e.Cancel = true; // 阻止開新視窗
            var browser = sender as WebBrowser;
            if (browser != null)
            {
                string url = browser.StatusText;
                if (!string.IsNullOrEmpty(url))
                {
                    browser.Navigate(url);
                }
            }
        }


    }


}
