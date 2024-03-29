# utf 8
# Compile Death and Poaf datasets

import numpy as np
import pandas as pd

import sys
sys.path.append('/home/guest/Yandex.Disk/GitHub/medstats/src/cxs')
import describe as descr


# +----------------------------------------------------------------------------


# Death

def DEATH():

    path = '/home/guest/Yandex.Disk/Документы/Документы/Сокольская/data/data_complete.xlsx'
    df = pd.read_excel(path, engine='openpyxl')
    # Clean values
    df = descr.columnn_normalizer(df, df.columns)

    # Make categorical Features
    cat_lst = [x for x in df.columns if len(set(df[x].dropna())) < 3 ]
    df = descr.column_categorical(df, cat_lst)

    # Select variables for X, y sets...

    X = df[[#'id',
    'коронарография (1-да, 0-нет)',
    'Возраст, лет',
    'Пол (Мужской/Женский - 1/0)',
    #'Масса тела (кг)',
    #'Рост, см',
    #'BSA',
    'ИМТ',
    'CG 1',
    #'MDRD 1',
    #'Kreatinin MGDL 1',
    #'креатинин 1',
    'ФВ ЛЖ(исх)',
    #'иКДР(исх)',
    'КДР(исх)',
    #'иКСР(исх)',
    'КСР(исх)',
    #'иКДО(исх)',
    'КДОисх (см3)',
    #'иКСО(исх)',
    'КСОисх (см3)',
    'Курение (Да/Нет - 1/0)',
    'АГ (Да/Нет - 1/0)',
    'Сахарный диабет (Да/Нет - 1/0)',
    'ХОБЛ (Да/Нет - 1/0)',
    'Онкозаболевание (Да/Нет - 1/0)',
    'Язвенная болезнь желудка/12пк (Да/Нет - 1/0)',
    'ХБП (Да/Нет - 1/0)',
    'Стеноз -1, недостаточность-0',
    'ИБС (Да/Нет - 1/0)',
    'Перенесенный ИМ (Да/Нет - 1/0)',
    # 'Аневризма ЛЖ (Да/Нет - 1/0)',
    # 'ДМПП (Да/Нет - 1/0)',
    # 'ДМЖП (Да/Нет - 1/0)',
    'ППС Порок МК (Да/Нет - 1/0)',
    'ППС Порок ТК (Да/Нет - 1/0)',
    'НРС д/о  (1-есть,0-нет)',
    'ФП д/о',
    'НРС ТП д/о(Да/Нет - 1/0)',
    'НРС ЖТ д/о(Да/Нет - 1/0)',
    'НРС НЖТ д/о(Да/Нет - 1/0)',
    'НРС ЧЖЭ д/о (Да/Нет - 1/0)',
    'НРС СССУ д/о (Да/Нет - 1/0)',
    'НРС АВ-блокады д/о (Да/Нет - 1/0)',
    'Легочная гипертензия (Да/Нет - 1/0)',
    'Атеросклероз БЦА (Да/Нет - 1/0)',
    'Атеросклероз артерий конечностей (Да/Нет - 1/0)',
    'Инсульт/ОНМК д/о(Да/Нет - 1/0)',
    # 'Патология аорты (Да/Нет - 1/0)',
    # 'Патология печени (Да/Нет - 1/0)',
    'Пиковый градиент на АК д/о',
    # 'Средний градиент на АК д/о',
    'ФК аортального клапана',
    #'площадь отверстия АоК ( AVA VTI)',
    # 'Объем ЛП (мл)',
    # 'Градиент давления на МК д/о',
    # 'Недостаточность на МКд/о (4/3/2/1/0 - 4/3/2/1/0)',
    #'Недостаточность на АК д/о (4/3/2/1/0 - 4/3/2/1/0)',
    #'Недостаточность на ТК д/о (4/3/2/1/0 - 4/3/2/1/0)',
    'Синусовый ритм д/о (Да/Нет - 1/0)',
    #'Ритм ЭКС (Да/Нет - 1/0)',
    #'Тахи-бради синдром (Да/Нет - 1/0)',
    'Атеросклероз (Да/Нет - 1/0)',
    # 'Тип кровоснабжения (левый/правый - 1/0)',
    # 'Бета-блокаторы д/о (Да/Нет - 1/0)',
    # 'и АПФд/о (Да/Нет - 1/0)',
    # 'АРА д/о(Да/Нет - 1/0)',
    # 'БКК дигидропиридиновые д/о (Да/Нет - 1/0)',
    # 'БКК недигидропиридиновые д/о(Да/Нет - 1/0)',
    # 'Статины д/о (Да/Нет - 1/0)',
    # 'АСК д/о(Да/Нет - 1/0)',
    # 'Клопидогрель д/о(Да/Нет - 1/0)',
    # 'Нитраты/аналогид/о (Да/Нет - 1/0)',
    #'Диуретики тиазидные д/о(Да/Нет - 1/0)',
    #'Диуретики петлевыед/о (Да/Нет - 1/0)',
    #'Диуретики калийсберегающие д/о (Да/Нет - 1/0)',
    #'Антикоагулянты д/о(АВК) (Да/Нет - 1/0)',
    #'Антикоагулянты (Прямые ингибиторы тромбина)д/о (Да/Нет - 1/0)',
    #'Антикоагулянты (ингибиторы фактора Ха)д/о (Да/Нет - 1/0)',
    #'Амиодарон д/о(Да/Нет - 1/0)',
    #'Антиаритмики д/о(1С класс) (Да/Нет - 1/0)',
    'Hb до операции',
    #'Hct до операции',
    #'тромбоциты до операции',
    #'Лей д/о',
    #'Нейтрофилы абс д/о',
    # 'Ней% д/о',
    # 'Glu д/о',
    # 'Фибриноген д/о',
    'Время ИК',
    'Время пережатия аорты',
    'АКШ',
    #'Стентирование (Да/Нет - 1/0)',
    'Протезирование МК (Да/Нет - 1/0)',
    'Пластика МК (Да/Нет - 1/0)',
    'Протезирование ТК (Да/Нет - 1/0)',
    'Пластика ТК (Да/Нет - 1/0)',
    'Лабиринт (Да/Нет - 1/0)',
    # 'РЧА (Да/Нет - 1/0)',
    # 'Сутки в ОРИТ',
    # 'Сутки в отделении после Оп',
    # 'Осложнения',
    'POAF',
    # 'НРС ТП п/о',
    # 'Восстановление СР',
    # 'Кардиоверсия',
    # 'ЭКС в п/о периоде (1-да, 0-нет)',
    # 'Перикардит',
    # 'СН',
    # 'Легочные осложнения ',
    # 'Surgical Priority Risk',
    # 'Risk of inhospital death',
    #'DEATH',
    'Метод кардиоплегии (1-Р,2-А,3-А+Р)_3.0',
    'Метод кардиоплегии (1-Р,2-А,3-А+Р)_2.0',
    'Метод кардиоплегии (1-Р,2-А,3-А+Р)_1.0'
    ]]

    # Save to disk
    X.to_pickle("X_death.pkl")

    y = df['DEATH']
    y.to_pickle("y_death.pkl")

    del X
    del y
    del df


def POAF():

    path = '/home/guest/Yandex.Disk/Документы/Документы/Сокольская/data/data_complete_poaf.xlsx'

    df = pd.read_excel(path, engine='openpyxl')

    df = descr.columnn_normalizer(df, df.columns)

    # Лист с категориальными переменными
    cat_lst = [x for x in df.columns if len(set(df[x].dropna())) < 3 ]

    df = descr.column_categorical(df, cat_lst)

    X = df[[#'id',
        'коронарография (1-да, 0-нет)',
        'Возраст, лет',
        'Пол (Мужской/Женский - 1/0)',
        #'Масса тела (кг)',
        #'Рост, см',
        #'BSA',
        'ИМТ',
        'CG 1',
        #'MDRD 1',
        #'Kreatinin MGDL 1',
        #'креатинин 1',
        'ФВ ЛЖ(исх)',
        #'иКДР(исх)',
        'КДР(исх)',
        #'иКСР(исх)',
        'КСР(исх)',
        #'иКДО(исх)',
        'КДОисх (см3)',
        #'иКСО(исх)',
        'КСОисх (см3)',
        'Курение (Да/Нет - 1/0)',
        'АГ (Да/Нет - 1/0)',
        'Сахарный диабет (Да/Нет - 1/0)',
        'ХОБЛ (Да/Нет - 1/0)',
        'Онкозаболевание (Да/Нет - 1/0)',
        'Язвенная болезнь желудка/12пк (Да/Нет - 1/0)',
        'ХБП (Да/Нет - 1/0)',
        'Стеноз -1, недостаточность-0',
        'ИБС (Да/Нет - 1/0)',
        'Перенесенный ИМ (Да/Нет - 1/0)',
        # 'Аневризма ЛЖ (Да/Нет - 1/0)',
        # 'ДМПП (Да/Нет - 1/0)',
        # 'ДМЖП (Да/Нет - 1/0)',
        # 'ППС Порок МК (Да/Нет - 1/0)',
        # 'ППС Порок ТК (Да/Нет - 1/0)',
        'НРС д/о  (1-есть,0-нет)',
        'ФП д/о',
        'НРС ТП д/о(Да/Нет - 1/0)',
        'НРС ЖТ д/о(Да/Нет - 1/0)',
        'НРС НЖТ д/о(Да/Нет - 1/0)',
        'НРС ЧЖЭ д/о (Да/Нет - 1/0)',
        'НРС СССУ д/о (Да/Нет - 1/0)',
        'НРС АВ-блокады д/о (Да/Нет - 1/0)',
        'Легочная гипертензия (Да/Нет - 1/0)',
        'Атеросклероз БЦА (Да/Нет - 1/0)',
        'Атеросклероз артерий конечностей (Да/Нет - 1/0)',
        'Инсульт/ОНМК д/о(Да/Нет - 1/0)',
        # 'Патология аорты (Да/Нет - 1/0)',
        # 'Патология печени (Да/Нет - 1/0)',
        'Пиковый градиент на АК д/о',
        # 'Средний градиент на АК д/о',
        'ФК аортального клапана',
        #'площадь отверстия АоК ( AVA VTI)',
        # 'Объем ЛП (мл)',
        # 'Градиент давления на МК д/о',
        # 'Недостаточность на МКд/о (4/3/2/1/0 - 4/3/2/1/0)',
        #'Недостаточность на АК д/о (4/3/2/1/0 - 4/3/2/1/0)',
        #'Недостаточность на ТК д/о (4/3/2/1/0 - 4/3/2/1/0)',
        'Синусовый ритм д/о (Да/Нет - 1/0)',
        #'Ритм ЭКС (Да/Нет - 1/0)',
        #'Тахи-бради синдром (Да/Нет - 1/0)',
        # 'Атеросклероз (Да/Нет - 1/0)',
        # 'Тип кровоснабжения (левый/правый - 1/0)',
         'Бета-блокаторы д/о (Да/Нет - 1/0)',
         'и АПФд/о (Да/Нет - 1/0)',
        # 'АРА д/о(Да/Нет - 1/0)',
        # 'БКК дигидропиридиновые д/о (Да/Нет - 1/0)',
        # 'БКК недигидропиридиновые д/о(Да/Нет - 1/0)',
         'Статины д/о (Да/Нет - 1/0)',
        # 'АСК д/о(Да/Нет - 1/0)',
        # 'Клопидогрель д/о(Да/Нет - 1/0)',
        # 'Нитраты/аналогид/о (Да/Нет - 1/0)',
        #'Диуретики тиазидные д/о(Да/Нет - 1/0)',
        #'Диуретики петлевыед/о (Да/Нет - 1/0)',
        #'Диуретики калийсберегающие д/о (Да/Нет - 1/0)',
        #'Антикоагулянты д/о(АВК) (Да/Нет - 1/0)',
        #'Антикоагулянты (Прямые ингибиторы тромбина)д/о (Да/Нет - 1/0)',
        #'Антикоагулянты (ингибиторы фактора Ха)д/о (Да/Нет - 1/0)',
        'Амиодарон д/о(Да/Нет - 1/0)',
        #'Антиаритмики д/о(1С класс) (Да/Нет - 1/0)',
        'Hb до операции',
        #'Hct до операции',
        #'тромбоциты до операции',
        #'Лей д/о',
        #'Нейтрофилы абс д/о',
        # 'Ней% д/о',
        # 'Glu д/о',
        # 'Фибриноген д/о',
        'Время ИК',
        'Время пережатия аорты',
        'АКШ',
        #'Стентирование (Да/Нет - 1/0)',
        'Протезирование МК (Да/Нет - 1/0)',
        'Пластика МК (Да/Нет - 1/0)',
        'Протезирование ТК (Да/Нет - 1/0)',
        'Пластика ТК (Да/Нет - 1/0)',
        'Лабиринт (Да/Нет - 1/0)',
        # 'РЧА (Да/Нет - 1/0)',
        'Сутки в ОРИТ',
        # 'Сутки в отделении после Оп',
        # 'Осложнения',
        # 'POAF',
        # 'НРС ТП п/о',
        # 'Восстановление СР',
        # 'Кардиоверсия',
        # 'ЭКС в п/о периоде (1-да, 0-нет)',
        'Перикардит',
        'СН',
        #'Легочные осложнения ',
        'Surgical Priority Risk',
        'Risk of inhospital death',
        #'DEATH',
        'Метод кардиоплегии (1-Р,2-А,3-А+Р)_3.0',
        'Метод кардиоплегии (1-Р,2-А,3-А+Р)_2.0',
        'Метод кардиоплегии (1-Р,2-А,3-А+Р)_1.0'
        ]]

        

    X.to_pickle("X_poaf.pkl")

    y = df['POAF']
    y.to_pickle("y_poaf.pkl")

    del X
    del y
    del df

if __name__ == '__main__':

    DEATH()
    POAF()

    print("Successful save...")