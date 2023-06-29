
# Normalize a list of data
# ------------------------
def ft_normalize(data: list) -> list:
    return [(elem - min(data)) / (max(data) - min(data)) for elem in data]