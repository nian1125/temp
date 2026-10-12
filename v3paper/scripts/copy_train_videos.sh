#!/usr/bin/env bash
# 拷贝 393 条训练数据用到的 189 个视频到 dockerdata
# 在新机器上执行: bash copy_train_videos.sh
set -euo pipefail

SRC_BASE="/mnt/250T_ceph/zhenjunchen/data/TimeLens-100K/video_shards"
DST_BASE="/dockerdata/zhenjunchen/train_videos"
mkdir -p "$DST_BASE"

copy_cat() {
  local cat="$1"; shift
  local src_dir="${SRC_BASE}/${cat}/${cat}"
  local dst_dir="${DST_BASE}/${cat}"
  mkdir -p "$dst_dir"
  for fn in "$@"; do
    if [ -f "${src_dir}/${fn}" ]; then
      cp -n "${src_dir}/${fn}" "${dst_dir}/${fn}" || echo "SKIP: $cat/$fn"
    else
      echo "MISS: $cat/$fn"
    fi
  done
}

echo "拷贝视频中 (189个, ~3G) ..."
echo ""

# cosmo_cap: 100 个
copy_cat cosmo_cap \
  -zdkTVpbN-g.mp4 0ELI_7GyxE8.mp4 0pwsGmSg5yk.mp4 0qadUj1c55o.mp4 \
  1JEeDNPvA4s.mp4 1R5iTNyx1ck.mp4 2P5I2kjP0gM.mp4 2Xl7DzEUxV4.mp4 \
  3SNW7_LbO-c.mp4 3dBn2L7On3k.mp4 4Db7PcfF5IE.mp4 56uTmAnOsBI.mp4 \
  5f1PvpH1nIg.mp4 5kcahKdPGEs.mp4 6d4d8yQmsUI.mp4 6geG1t4CYbs.mp4 \
  77Ra4I4ViKg.mp4 7I1t5vEMNVo.mp4 85_QSCFn4dQ.mp4 8O4qBuN_SwU.mp4 \
  8Sl8kJ2_r3o.mp4 8XhaJSSQ-mg.mp4 9Bx6H1mPmBA.mp4 9Yj8WxcKQB0.mp4 \
  A0ppXhJcs40.mp4 AEW4I94M1BY.mp4 AMn9OoNxZDE.mp4 BLQjOWe-3eA.mp4 \
  B_d0zNSIqRU.mp4 COaSBkUP9sY.mp4 CckzXqMuwTk.mp4 FINgEiMf6kU.mp4 \
  FseRYiR-gC0.mp4 FsoXUbmDBu8.mp4 GBCY7K6aTgw.mp4 GGBgwBR_dZ4.mp4 \
  GUYzB3KxRtg.mp4 IcEPAG3DG5I.mp4 Iuqm1yKeVls.mp4 JwxJ5M3uLLg.mp4 \
  KAI_BNhMWFw.mp4 KB5GJJSTOwA.mp4 LsO-5HZLBrc.mp4 M38GkCjlu_w.mp4 \
  M_4cPGqZsvA.mp4 OLR7SNb9GSM.mp4 O_fVnQpQrcA.mp4 P4nsa5f7PcU.mp4 \
  P_H0Pk_28NE.mp4 QcQAzFoV7oI.mp4 QCKFL8AOP6k.mp4 QHZ4CY7aABc.mp4 \
  QSWdN30W8vo.mp4 QfFZp0O6n1k.mp4 RX7fjGr-24M.mp4 RVB6iH5_fYU.mp4 \
  SUGZ4KAGqPY.mp4 TcSdvfNpVQw.mp4 UyZzJFHyod4.mp4 Um0cYAxdz5c.mp4 \
  V6tMP_NQHQ8.mp4 VAAqKia_MDM.mp4 VNY82MKYt5k.mp4 VwK_EL3zgtM.mp4 \
  Vxdy1HizFao.mp4 Vu8qFZ6rmGc.mp4 W3Hm0t6NXOU.mp4 WCEQzP1sS6I.mp4 \
  WX5UprRR0uY.mp4 WbnJcIKz8kQ.mp4 WqYat2zJZy4.mp4 XbhG2b3QYak.mp4 \
  Y01ir5XpZfk.mp4 ZAPK8ekb7Xk.mp4 ZG2JMnDz5as.mp4 ZGJrHVBVJ6E.mp4 \
  _Ins-KtuYwo.mp4 agI2X3aQmAY.mp4 b3hBT7zrdDI.mp4 bwA7ZbAn3OM.mp4 \
  dW8DYMCjC9E.mp4 glgb6WM9aQs.mp4 iPjllu51UMk.mp4 jrL8CAsBQQ8.mp4 \
  ks0n_PhJSKI.mp4 mRRFe6ZTUvg.mp4 qYpUR99Bnyk.mp4 sDYgFtpDY-A.mp4 mDFI9A3JmCY.mp4

echo "cosmo_cap 完成"

# didemo: 50 个
copy_cat didemo \
  105508733@N04_10716104373_e77a2b3434.mp4 10770768@N08_3720316878_56e4c0d3bd.mp4 \
  11357767@N00_3242573802_0f4d953ec6.mp4 21628929@N08_5695807437_ac43a9d8e1.mp4 \
  22651740@N00_3860201214_feb33344fd.mp4 28536568@N00_5089241286_bdef8c7995.mp4 \
  29485173@N03_4576115181_a3e5328d22.mp4 34392764@N03_4831480037_aa76430e26.mp4 \
  34418903@N00_5543596024_7dd6d1f037.mp4 35034347347@N01_3158805392_085d51e2b7.mp4 \
  35034347347@N01_3317851426_daba5688b7.mp4 35367802@N03_3401381397_83a4b1b0e3.mp4 \
  35776264@N08_5366455914_f1a06a9675.mp4 36135889@N00_4649539359_bdf1c8e92c.mp4 \
  38011706@N00_8222395976_e6b2032175.mp4 38928586@N07_4440186396_5c57d4ab76.mp4 \
  39106725@N00_3101999185_823b3e1abd.mp4 40457523@N00_7289251572_dc2c50694c.mp4 \
  44568159@N04_3407679713_319dc42432.mp4 47424495@N00_5577812023_24ab7afe59.mp4 \
  49673251@N00_3887996284_4c0e72c895.mp4 49768352@N03_4982853059_bf3ed86f57.mp4 \
  51269448@N06_7707725118_a54b233d80.mp4 52087980@N00_1549354541_0ddd67a613.mp4 \
  53535391@N00_3375211386_2360ab23c9.mp4 54831387@N00_9375971215_9f9dac02e7.mp4 \
  57210687@N06_7279764536_12140235b4.mp4 61635869@N00_9519029907_91e7f9e01f.mp4 \
  64424124@N07_2485126379_1f45b09125.mp4 7368666@N03_3879315419_b154fa6ebb.mp4 \
  77987497@N00_6381765245_228d2d661c.mp4 78495119@N00_4512063081_c98096b0eb.mp4 \
  7882602@N08_3496833846_845be2b9dd.mp4 80751277@N00_4160598366_81aa7045b4.mp4 \
  82779205@N00_6866179185_e67e5b42fe.mp4 87719130@N00_6949799723_7d5e7cafb0.mp4 \
  8849821@N04_5193863515_c01f4e5a27.mp4 93366149@N00_5480001103_e8239956dc.mp4 \
  93444230@N00_8299647002_95aeb5eb34.mp4 93523735@N00_2625102723_02e1a05a2b.mp4 \
  94083810@N00_3957829688_4a6bae3383.mp4 95843739@N00_4409860247_8e8f78f2dc.mp4 \
  96076528@N00_9433112315_6862be263c.mp4 99238404@N00_3421078789_42a840ddd7.mp4 \
  ekumbrella.mp4 38815827@N00_4948468327_7c6a8b4af1.mp4

echo "didemo 完成"

# internvid_vtime: 22 个
copy_cat internvid_vtime \
  2bZqayGS1LM.mp4 49vLAWO5Vhs.mp4 4rVjWpahnW8.mp4 7DmrOAcNp_E.mp4 \
  Bau3mEqr66M.mp4 C0Fs14lEEgA.mp4 D3l8v3WtViU.mp4 DYwR-dbFHMg.mp4 \
  Go6WCAXKfBk.mp4 IsHhBY7RjsU.mp4 IwinuW1NwpA.mp4 KSAgyA4gGxo.mp4 \
  MmcWC_-dObk.mp4 Rsfsn4LFsuk.mp4 Sv1zMjyYiRk.mp4 U2_21m11tCg.mp4 \
  V3o1yD-B8rI.mp4 ZP4JqLD9ujg.mp4 j9sEf5Z4pXQ.mp4 nHRFYvQONQ0.mp4 \
  uc9RQ5jQ4sg.mp4 yw62K7rLgZo.mp4

echo "internvid_vtime 完成"

# hirest: 10 个
copy_cat hirest \
  5BhYKtFlhVo.mp4 9w0B74ZwfRA.mp4 FQIJK-sJRI8.mp4 Opx-62Yt8mg.mp4 \
  QO0WlgkGc5E.mp4 RGvzmRyJwM4.mp4 VP9wvE8IsgU.mp4 jf0Sc3C0_LQ.mp4 \
  nyTFNskBTpo.mp4 vn0BdCYAu3U.mp4

echo "hirest 完成"

# queryd: 7 个
copy_cat queryd \
  DF8_5mGdcq4.mp4 LJosiEHwWxc.mp4 MmVXLM97YL0.mp4 S9DdQ0n9M1s.mp4 \
  TfM24bNMGoE.mp4 jEIrT1DNLlc.mp4 sWQZXQ3KqQU.mp4

echo "queryd 完成"

echo ""
echo "=========================================="
echo "全部完成! 验证:"
echo "=========================================="
for cat in cosmo_cap didemo internvid_vtime queryd hirest; do
  count=$(ls /dockerdata/zhenjunchen/train_videos/${cat}/ 2>/dev/null | wc -l)
  echo "  ${cat}: ${count} 个"
done
total=$(find /dockerdata/zhenjunchen/train_videos/ -type f -name "*.mp4" 2>/dev/null | wc -l)
echo "  总计: ${total} 个"
echo "  大小: $(du -sh /dockerdata/zhenjunchen/train_videos/ 2>/dev/null | cut -f1)"
