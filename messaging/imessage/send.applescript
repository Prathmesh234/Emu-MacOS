-- messaging/imessage/send.applescript
-- Send an iMessage to a specified buddy ("phone or email") via Messages.app.
--
-- Invocation (from Node):
--   osascript send.applescript "+15551234567" "hello from emu"
--
-- We never use `osascript -e` with a constructed string — that would mean
-- splicing untrusted text into source. Passing both arguments through argv
-- keeps the AppleScript source itself static and reviewable.

on run argv
    if (count of argv) < 2 then
        return "ERR missing args"
    end if
    set targetHandle to item 1 of argv
    set messageBody to item 2 of argv

    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        try
            set targetBuddy to buddy targetHandle of targetService
        on error errMsg
            return "ERR no-buddy: " & errMsg
        end try
        try
            send messageBody to targetBuddy
            return "OK"
        on error errMsg
            return "ERR send-failed: " & errMsg
        end try
    end tell
end run
