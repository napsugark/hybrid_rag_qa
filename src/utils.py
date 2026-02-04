def format_time(seconds):
    """Format seconds as Xm YY.YYs or YY.YYs"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"
