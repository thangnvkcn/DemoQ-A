
synonym = {"Chương trình bằng kép": ["chuong trinh bang kep", "ctr bằng kép", "chtr bang kep","ctr bang kep","ctr bằng kep","chương trình bằng kep","chuong trinh bằng kép"],
                   "Ngôn Ngữ Hàn": ["nn Hàn", "ngon ngu Han", "nn Han","ngôn ngữ Han","ngon ngu Hàn"],
                   "Ngôn ngữ Nhật": ["nn Nhật", "ngon ngu Nhat", "nn Nhat","ngôn ngữ Nhat","ngon ngu Nhật"],
                   "Ngôn ngữ Anh": ["nn Anh", "ngôn ngữ anh", "ngon ngu Anh", "ngon ngữ Anh","ngôn ngu Anh"],
                   "Luật kinh doanh": ["luật kd", "luat kinh doanh", "ngành lkd"],
                   "chương trình đại học do Đại học Quốc gia Hà Nội cấp bằng": ["ctr dhqg cấp bằng", "ctr đhqg cap bang"],
                   "kinh doanh quốc tế": ["kdqt", "kd qte", "kinh doanh quoc te"],
                   "tin học và kĩ thuật máy tính": ["thktmt", "tin hoc ktmt", "tin hoc ky thuat may tinh"],
                   "hệ thống thông tin": ["httt", "he thong thong tin", "he thong tt", "hệ thống tt"],
                   "Ngành Kế toán, Phân tích và Kiểm Toán": ["ctr AC", "ktptkt", "ctr kế kiểm", "ctr ke kiem", "chuong trinh ke kiem"],
                   "Dữ liệu và phân tích kinh doanh": ["dba", "du lieu va phan tich kinh doanh", "phan tich du lieu kd", "phân tích dữ liệu kinh doanh"],
                   "Chương trình liên kết": ["ctr lk", "ctlk", "chuong trinh lien ket"],
                   "kế toán và tài chính": ["ke toan tai chinh", "ke toan va tai chinh", "kt va tc", "kttc"],
                   "quản trị khách sạn thể thao du lịch": ["host", "quan tri ks the thao du lich", "quan tri ksttdl"],
                   "khoa học quản lý": ["khql", "khoa hoc quan ly"],
                   "chương trình kế toán": ["ctr ke toan", "chuong trinh ke toan"],
                   "marketing": ["marketting"]}


def replaceABwithC(input, pattern, replaceWith):
  return input.replace(pattern, replaceWith)

def find_in_dictionary(str):
    for keys,values in zip(synonym.keys(),synonym.values()):
      for value in values:
        if value.lower() in str.lower():
          str = replaceABwithC(str,value,keys)
    return str
