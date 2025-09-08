using System.ComponentModel;
using System.Diagnostics;
using Microsoft.Win32;
using System.Drawing.Imaging;


namespace FlashGameLoader
{
    public class Program : Form
    {
        private WebBrowser webBrowser;
        private Label statusLabel;

        // 設定瀏覽器IE的版本為IE11，設定完才可以用CSS的方式修改網頁的縮放，預設的IE7沒辦法用CSS
        private static void SetBrowserFeatureControl()
        {
            string appName = System.IO.Path.GetFileName(Process.GetCurrentProcess().MainModule.FileName);
            using (var key = Registry.CurrentUser.CreateSubKey(
                @"Software\Microsoft\Internet Explorer\Main\FeatureControl\FEATURE_BROWSER_EMULATION",
                RegistryKeyPermissionCheck.ReadWriteSubTree))
            {
                key.SetValue(appName, 8000, RegistryValueKind.DWord); // 11001 = IE11 edge mode
            }
        }

        [STAThread]
        public static void Main()
        {
            SetBrowserFeatureControl();
            Application.EnableVisualStyles();
            Application.Run(new Program());
        }

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
                webBrowser.Document.InvokeScript("execScript", new object[]
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
                if (webBrowser.Document != null && webBrowser.Url.AbsoluteUri.Contains("/Login"))
                {
                    // 網頁縮放比例設定為80%，用來方便截圖
                    if (webBrowser.Document?.Body != null)
                    {
                        webBrowser.Document.Body.Style = "zoom:80%;";
                    }

                    // 自動輸入帳號
                    var userBox = webBrowser.Document.GetElementById("UserID");
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
                    // 自動輸入密碼
                    var captcha = webBrowser.Document.GetElementById("CheckText");
                    if (captcha != null)
                    {
                        captcha.SetAttribute("value", "差一滴滴");
                    }

                    // 自動擷取驗證碼圖片
                    AutoCaptureVerifyCodeImage();
                }
                // 用JS把左邊的資訊欄刪除，不刪除會影響到畫面顯示
                if (webBrowser.Url != null && webBrowser.Url.AbsoluteUri.Contains("/Game/Server/"))
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
        private void AutoCaptureVerifyCodeImage()
        {
            // 使用Timer延遲執行，確保圖片已完全載入
            System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
            timer.Interval = 1000; // 延遲1秒
            timer.Tick += (sender, e) =>
            {
                timer.Stop();
                timer.Dispose();
                CaptureVerifyCodeImage();
            };
            timer.Start();
        }
        private void CaptureVerifyCodeImage()
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

                object result = webBrowser.Document.InvokeScript("eval", new object[] { script });
                string base64Data = result?.ToString();

                if (base64Data == "loading")
                {
                    MessageBox.Show("圖片正在載入中，請稍後再試");
                    return;
                }

                if (!string.IsNullOrEmpty(base64Data) && base64Data.StartsWith("data:image"))
                {
                    // 移除data:image/png;base64,前綴
                    string base64String = base64Data.Substring(base64Data.IndexOf(',') + 1);
                    byte[] imageBytes = Convert.FromBase64String(base64String);

                    string folder = Path.Combine(Application.StartupPath, "captcha");

                    // 如果資料夾不存在就建立
                    if (!Directory.Exists(folder))
                    {
                        Directory.CreateDirectory(folder);
                    }

                    // 儲存圖片
                    string fileName = $"verifycode_{DateTime.Now:yyyyMMdd_HHmmss}.png";
                    string filePath = Path.Combine(folder, fileName);
                    File.WriteAllBytes(filePath, imageBytes);
                    //
                    // MessageBox.Show($"驗證碼圖片已儲存至: {filePath}");
                    UpdateStatus($"驗證碼圖片已儲存至: {filePath}");
                }

            }
            catch (Exception ex)
            {
                MessageBox.Show($"擷取驗證碼圖片失敗: {ex.Message}");
                // 如果所有方法都失敗，至少截取整頁
            }
        }

        // 新增：重新整理驗證碼圖片的方法
        private void RefreshVerifyCodeImage()
        {
            try
            {
                if (webBrowser.Document == null)
                {
                    UpdateStatus("網頁尚未載入");
                    return;
                }

                var img = webBrowser.Document.GetElementById("verifyCodeImg");
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