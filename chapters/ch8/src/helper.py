def tokenize(text):
    tokens = text.lower().split()
    return tokens

def create_word_dictionary(word_list):
    """
    빈 딕셔너리를 생성
    """
    word_dict = {
        "PAD": 0,  # 패딩 토큰을 위한 인덱스
        "UNK": 1,   # 알 수 없는 단어를 위한 인덱스
    }
    # 고유한 값을 카운트
    counter = 2

    # 리스트를 순회하면서 고유한 단어에 인덱스를 할당
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = counter
            counter += 1

    return word_dict

def text_to_sequence(sentence, word_dict):
    # 문장을 소문자로 바꾸고 단어로 분할
    words = sentence.lower().strip().split()

    # 각 단어를 해당 숫자로 변환
    number_sequence = [
    word_dict.get(word, word_dict["UNK"])
    for word in words]

    return number_sequence

def pad_sequence(sequences, max_length=None):
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    # 각 시퀀스의 앞에 0으로 패딩한다.
    padded_sequences = []
    for seq in sequences:
        # 필요한 0 개수를 계산
        num_zeros = max_length - len(seq)
        # 패딩된 시퀀스를 만든다.
        padded_seq = [0] * num_zeros + list(seq)
        padded_sequences.append(padded_seq)

    return padded_sequences

def split_sequences(sequences):
    # 각 시퀀스에서 마지막 원소를 제거하여 xs를 생성
    xs = [seq[:-1] for seq in sequences]
    # 각 시퀀스의 마지막 원소를 사용하여 레이블을 만든다.
    labels = [seq[-1:] for seq in sequences]
    # [-1:]로 하면 하나의 원소를 가진 리스트가 만들어진다.
    # 리스트 대신에 하나의 숫자를 레이블로 만들려면 다음처럼 하면된다.
    # labels = [seq[-1] for seq in sequnces]
    return xs, labels

def one_hot_encode_with_checks(value, corpus_size):
    # 값이 적절한 범위 안에 있는지 확인
    if not 0 <= value < corpus_size:
        raise ValueError(f"{value}는 어휘 사전 크기 {corpus_size}의 범위를 벗어납니다.")
    # 원-핫 인코딩 벡터 생성
    encoded = [0] * corpus_size
    encoded[value] = 1
    return encoded

